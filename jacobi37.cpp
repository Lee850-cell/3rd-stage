#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>  // 新增：文件輸出

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int totalsize = 16;
    const int mysize = totalsize / 4; // 每個進程處理 4 列
    const int steps = 10;
    
    // 定義數組：a[列][行] 為了保證每一列數據在內存中是連續的
    float a[mysize + 2][totalsize];
    float b[mysize + 2][totalsize];

    // 1. 初始化數組
    for (int j = 0; j < mysize + 2; ++j) {
        for (int i = 0; i < totalsize; ++i) {
            a[j][i] = 0.0f;
            // 頂部和底部邊界賦初值 8.0
            if (i == 0 || i == totalsize - 1) a[j][i] = 8.0f;
        }
    }

    // 左右兩端進程的物理邊界賦初值
    if (myid == 0) for (int i = 0; i < totalsize; ++i) a[1][i] = 8.0f;
    if (myid == 3) for (int i = 0; i < totalsize; ++i) a[mysize][i] = 8.0f;

    // 2. 確定鄰居與迭代區間
    int left = (myid > 0) ? myid - 1 : MPI_PROC_NULL;
    int right = (myid < 3) ? myid + 1 : MPI_PROC_NULL;

    int begin_col = 1;
    int end_col = mysize;
    if (myid == 0) begin_col = 2;
    if (myid == 3) end_col = mysize - 1;

    // 3. --- 初始化持久通信 (Persistent Requests) ---
    MPI_Request req[4];
    MPI_Send_init(&b[end_col][0], totalsize, MPI_FLOAT, right, 3, MPI_COMM_WORLD, &req[0]);
    MPI_Send_init(&b[begin_col][0], totalsize, MPI_FLOAT, left, 4, MPI_COMM_WORLD, &req[1]);
    MPI_Recv_init(&a[0][0], totalsize, MPI_FLOAT, left, 3, MPI_COMM_WORLD, &req[2]);
    MPI_Recv_init(&a[mysize + 1][0], totalsize, MPI_FLOAT, right, 4, MPI_COMM_WORLD, &req[3]);

    // 4. --- 執行迭代 ---
    for (int n = 0; n < steps; ++n) {
        // A. 計算邊界部分
        for (int i = 1; i < totalsize - 1; ++i) {
            b[begin_col][i] = (a[begin_col+1][i] + a[begin_col-1][i] + 
                               a[begin_col][i+1] + a[begin_col][i-1]) * 0.25f;
            b[end_col][i]   = (a[end_col+1][i] + a[end_col-1][i] + 
                               a[end_col][i+1] + a[end_col][i-1]) * 0.25f;
        }

        // B. 激活所有持久通信
        MPI_Startall(4, req);

        // C. 計算中間不依賴鄰居的部分 (Overlap 計算與通信)
        for (int j = begin_col + 1; j < end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) {
                b[j][i] = (a[j+1][i] + a[j-1][i] + a[j][i+1] + a[j][i-1]) * 0.25f;
            }
        }

        // D. 更新數組
        for (int j = begin_col; j <= end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) a[j][i] = b[j][i];
        }

        // E. 等待本輪通信完成
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // ===========================================================
    // 修改部分：收集所有數據並寫入 txt 文件
    // ===========================================================
    
    // 5. 準備本地數據（去掉影子格，只取真正計算的部分）
    std::vector<float> local_data;
    // 注意：每個進程的列數是 (end_col - begin_col + 1)
    int local_cols = end_col - begin_col + 1;
    local_data.reserve(local_cols * (totalsize - 2));
    
    for (int i = 1; i < totalsize - 1; ++i) {        // 行：1 到 14
        for (int j = begin_col; j <= end_col; ++j) {  // 列：每個進程負責的範圍
            local_data.push_back(a[j][i]);
        }
    }

    if (myid == 0) {
        // --- Rank 0 的工作：創建完整矩陣並收集中碎片 ---
        std::vector<float> full_matrix((totalsize - 2) * totalsize, 0.0f);
        
        // A. 先把 Rank 0 自己的數據填進去
        int my_cols = end_col - begin_col + 1;
        int my_start_col = 0;  // Rank 0 從第 0 列開始
        
        for (int i = 0; i < totalsize - 2; ++i) {
            for (int j = 0; j < my_cols; ++j) {
                full_matrix[i * totalsize + my_start_col + j] = local_data[i * my_cols + j];
            }
        }
        
        // B. 接收來自其他進程的數據
        for (int p = 1; p < numprocs; ++p) {
            // 計算這個進程負責的列範圍
            int p_begin_col = (p == 0) ? 1 : (p == 1) ? 1 : (p == 2) ? 1 : 1;
            int p_end_col = (p == 0) ? mysize : (p == 1) ? mysize : (p == 2) ? mysize : mysize;
            if (p == 0) p_begin_col = 2;
            if (p == 3) p_end_col = mysize - 1;
            
            int p_cols = p_end_col - p_begin_col + 1;
            int p_start_col = p * mysize;  // Rank 1 從第 4 列開始，Rank 2 從第 8 列，Rank 3 從第 12 列
            if (p == 0) p_start_col = 0;
            
            std::vector<float> recv_buf((totalsize - 2) * p_cols);
            MPI_Recv(recv_buf.data(), (totalsize - 2) * p_cols, MPI_FLOAT, p, 99, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < totalsize - 2; ++i) {
                for (int j = 0; j < p_cols; ++j) {
                    full_matrix[i * totalsize + p_start_col + j] = recv_buf[i * p_cols + j];
                }
            }
        }
        
        // C. 寫入文件
        std::cout << "Writing to result.txt..." << std::endl;
        std::ofstream outfile("result.txt");
        if (outfile.is_open()) {
            // 寫入列標題
            outfile << "    ";
            for (int j = 0; j < totalsize; ++j) {
                outfile << std::setw(8) << "Col" << j+1;
            }
            outfile << std::endl;
            
            // 寫入數據
            for (int i = 0; i < totalsize - 2; ++i) {
                outfile << "Row" << std::setw(3) << i+1 << ":";
                for (int j = 0; j < totalsize; ++j) {
                    outfile << std::fixed << std::setprecision(2) 
                            << std::setw(8) << full_matrix[i * totalsize + j];
                }
                outfile << std::endl;
            }
            outfile.close();
            std::cout << "Done! File saved as result.txt" << std::endl;
        } else {
            std::cerr << "Error: Cannot open file!" << std::endl;
        }
        
    } else {
        // --- 其他 Rank 的工作：把自己的數據發給 Rank 0 ---
        int local_cols = end_col - begin_col + 1;
        MPI_Send(local_data.data(), (totalsize - 2) * local_cols, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
    }

    // 6. 釋放持久通信句柄
    for (int i = 0; i < 4; ++i) MPI_Request_free(&req[i]);

    MPI_Finalize();
    return 0;
}