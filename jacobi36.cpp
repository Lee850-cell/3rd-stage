#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows = 16;
    const int mysize = 4; // 每個進程處理的列數
    const int steps = 10;
    
    // a[列][行]：為了讓一列數據在內存中連續，方便 MPI 發送
    std::vector<std::vector<float>> a(mysize + 2, std::vector<float>(rows, 0.0f));
    std::vector<std::vector<float>> b(mysize + 2, std::vector<float>(rows, 0.0f));

    // 1. 初始化邊界條件
    for (int i = 0; i < rows; ++i) {
        a[0][i] = 8.0f;           // 左邊界（虛擬列）
        a[mysize + 1][i] = 8.0f; // 右邊界（虛擬列）
    }
    
    // 頂部和底部邊界
    for (int j = 0; j < mysize + 2; ++j) {
        a[j][0] = 8.0f;           // 上邊界
        a[j][rows - 1] = 8.0f;    // 下邊界
    }
    
    // 左右兩端進程的物理邊界
    if (rank == 0) {
        for (int i = 0; i < rows; ++i) a[1][i] = 8.0f;
    }
    if (rank == 3) {
        for (int i = 0; i < rows; ++i) a[mysize][i] = 8.0f;
    }

    // 2. 確定鄰居
    int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < 3) ? rank + 1 : MPI_PROC_NULL;

    int begin_col = (rank == 0) ? 2 : 1;
    int end_col = (rank == 3) ? mysize - 1 : mysize;
    int local_cols = end_col - begin_col + 1;  // 每個進程實際計算的列數

    // 3. 迭代計算
    for (int n = 0; n < steps; ++n) {
        // A. 計算邊界列 (依賴外部數據的部分)
        for (int i = 1; i < rows - 1; ++i) {
            // 計算最左邊界列
            b[1][i] = (a[0][i] + a[2][i] + a[1][i-1] + a[1][i+1]) * 0.25f;
            // 計算最右邊界列
            b[mysize][i] = (a[mysize-1][i] + a[mysize+1][i] + a[mysize][i-1] + a[mysize][i+1]) * 0.25f;
        }

        // B. 非阻塞通訊：交換邊界
        MPI_Request reqs[4];
        // 發送給右邊，接收來自左邊
        MPI_Isend(&b[mysize][0], rows, MPI_FLOAT, right, 3, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&a[0][0], rows, MPI_FLOAT, left, 3, MPI_COMM_WORLD, &reqs[1]);
        // 發送給左邊，接收來自右邊
        MPI_Isend(&b[1][0], rows, MPI_FLOAT, left, 4, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&a[mysize+1][0], rows, MPI_FLOAT, right, 4, MPI_COMM_WORLD, &reqs[3]);

        // C. 計算中間不需要通訊的部分 (重疊 Overlap)
        for (int j = 2; j < mysize; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                b[j][i] = (a[j-1][i] + a[j+1][i] + a[j][i-1] + a[j][i+1]) * 0.25f;
            }
        }

        // D. 更新數組
        for (int j = 1; j <= mysize; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                a[j][i] = b[j][i];
            }
        }

        // E. 完成所有通訊，確保下一輪開始前所有數據到位
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }

    // ===========================================================
    // 結果輸出部分
    // ===========================================================
    
    // 4. 準備本地數據（去掉影子格，只取真正計算的部分）
    std::vector<float> local_data;
    local_data.reserve(local_cols * (rows - 2));
    
    for (int i = 1; i < rows - 1; ++i) {        // 行：1 到 14
        for (int j = begin_col; j <= end_col; ++j) {  // 列：每個進程負責的範圍
            local_data.push_back(a[j][i]);
        }
    }

    if (rank == 0) {
        // --- Rank 0 的工作：創建完整矩陣並收集中碎片 ---
        const int total_rows = rows - 2;      // 14 行
        const int total_cols = rows;           // 16 列
        std::vector<float> full_matrix(total_rows * total_cols, 0.0f);
        
        // A. 先把 Rank 0 自己的數據填進去
        int my_start_col = 0;  // Rank 0 從第 0 列開始
        int my_cols = end_col - begin_col + 1;
        
        for (int i = 0; i < total_rows; ++i) {
            for (int j = 0; j < my_cols; ++j) {
                full_matrix[i * total_cols + my_start_col + j] = local_data[i * my_cols + j];
            }
        }
        
        // B. 接收來自其他進程的數據
        for (int p = 1; p < size; ++p) {
            // 計算這個進程的列範圍
            int p_begin_col, p_end_col, p_start_col;
            
            if (p == 0) {
                p_begin_col = 2;
                p_end_col = mysize;
                p_start_col = 0;
            } else if (p == 1) {
                p_begin_col = 1;
                p_end_col = mysize;
                p_start_col = mysize;  // 從第 4 列開始
            } else if (p == 2) {
                p_begin_col = 1;
                p_end_col = mysize;
                p_start_col = mysize * 2;  // 從第 8 列開始
            } else {  // p == 3
                p_begin_col = 1;
                p_end_col = mysize - 1;
                p_start_col = mysize * 3;  // 從第 12 列開始
            }
            
            int p_cols = p_end_col - p_begin_col + 1;
            
            std::vector<float> recv_buf(total_rows * p_cols);
            MPI_Recv(recv_buf.data(), total_rows * p_cols, MPI_FLOAT, p, 99, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < p_cols; ++j) {
                    full_matrix[i * total_cols + p_start_col + j] = recv_buf[i * p_cols + j];
                }
            }
        }
        
        // C. 寫入文件
        std::cout << "Writing to result.txt..." << std::endl;
        std::ofstream outfile("result_nonblock.txt");
        if (outfile.is_open()) {
            // 寫入列標題
            outfile << "    ";
            for (int j = 0; j < total_cols; ++j) {
                outfile << std::setw(8) << "Col" << j+1;
            }
            outfile << std::endl;
            outfile << "    ";
            for (int j = 0; j < total_cols; ++j) {
                outfile << std::setw(8) << "----";
            }
            outfile << std::endl;
            
            // 寫入數據
            for (int i = 0; i < total_rows; ++i) {
                outfile << "Row" << std::setw(3) << i+1 << ":";
                for (int j = 0; j < total_cols; ++j) {
                    outfile << std::fixed << std::setprecision(2) 
                            << std::setw(8) << full_matrix[i * total_cols + j];
                }
                outfile << std::endl;
            }
            outfile.close();
            std::cout << "Done! File saved as result_nonblock.txt" << std::endl;
            
            // 同時輸出到螢幕（可選）
            std::cout << "\nTemperature distribution after " << steps << " iterations:" << std::endl;
            std::cout << "    ";
            for (int j = 0; j < total_cols; ++j) {
                std::cout << std::setw(8) << "Col" << j+1;
            }
            std::cout << std::endl;
            for (int i = 0; i < total_rows; ++i) {
                std::cout << "Row" << std::setw(3) << i+1 << ":";
                for (int j = 0; j < total_cols; ++j) {
                    std::cout << std::fixed << std::setprecision(2) 
                              << std::setw(8) << full_matrix[i * total_cols + j];
                }
                std::cout << std::endl;
            }
        } else {
            std::cerr << "Error: Cannot open file!" << std::endl;
        }
        
    } else {
        // --- 其他 Rank 的工作：把自己的數據發給 Rank 0 ---
        int local_cols = end_col - begin_col + 1;
        MPI_Send(local_data.data(), (rows - 2) * local_cols, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}