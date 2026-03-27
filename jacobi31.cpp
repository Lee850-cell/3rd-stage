#include <mpi.h>
#include <iostream>
#include <vector>



int main(int argc, char** argv) {
    // 1. 初始化環境
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 本程序僅支援 2 個進程
    if (size != 2) {
        if (rank == 0) std::cerr << "Error: This program requires exactly 2 processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int TAG = 99;
    int data_buffer = 0;
    MPI_Request request;
    MPI_Status status;

    if (rank == 0) {
        // --- 傳送端 (Rank 0) ---
        data_buffer = 42;
        std::cout << "[Rank 0] Sending data: " << data_buffer << std::endl;
        
        // 使用標準發送
        MPI_Send(&data_buffer, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD);
        std::cout << "[Rank 0] Data sent.\n";

    } else if (rank == 1) {
        // --- 接收端 (Rank 1) ---
        
        // 1. 發起非阻塞接收
        MPI_Irecv(&data_buffer, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &request);
        std::cout << "[Rank 1] Irecv posted. Now attempting to cancel...\n";

        // 2. 立即嘗試取消通訊
        MPI_Cancel(&request);

        // 3. 重要：即便取消了，也必須執行 Wait 或 Test 來釋放 Request 資源
        MPI_Wait(&request, &status);

        // 4. 檢查取消是否成功
        int cancelled_flag = 0;
        MPI_Test_cancelled(&status, &cancelled_flag);

        if (cancelled_flag) {
            std::cout << "[Rank 1] Communication SUCCESSFULLY cancelled. Buffer is unchanged.\n";
            
            // 按照你之前的邏輯：如果取消成功，則重新發起接收
            std::cout << "[Rank 1] Re-posting Irecv because we still need the data...\n";
            MPI_Irecv(&data_buffer, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            std::cout << "[Rank 1] Final data received: " << data_buffer << std::endl;
        } else {
            // 如果取消失敗，說明數據在取消動作完成前就已經抵達了
            std::cout << "[Rank 1] Communication CANCEL FAILED. Data already arrived!\n";
            std::cout << "[Rank 1] Data received: " << data_buffer << std::endl;
        }
    }

    // 5. 結束環境
    MPI_Finalize();
    return 0;
}