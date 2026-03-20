#include <mpi.h>
#include <iostream>
#include <vector>

void test_rsend() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "This test requires exactly 2 processes!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int TEST_SIZE = 2000;
    const int TAG = 1456;
    const int COUNT = TEST_SIZE / 3;

    std::vector<float> send_buf(TEST_SIZE, 0.0f);
    std::vector<float> recv_buf(TEST_SIZE, 0.0f);
    MPI_Status status;
    MPI_Request request;

    if (rank == 0) {
        // --- 傳送端 (Rank 0) ---
        std::cout << "[Rank 0] Waiting for 'Ready' signal from Rank 1..." << std::endl;

        // 1. 阻塞接收一個 0 長度的消息，作為「信號彈」
        // 使用 MPI_INT 和 0 數量，這只是一個同步機制
        MPI_Recv(nullptr, 0, MPI_INT, 1, TAG, MPI_COMM_WORLD, &status);

        // 2. 確定 Rank 1 已經準備好了，現在執行就緒發送
        std::cout << "[Rank 0] Signal received. Posting Ready Send (MPI_Rsend)..." << std::endl;
        MPI_Rsend(send_buf.data(), COUNT, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD);

    } else {
        // --- 接收端 (Rank 1) ---
        std::cout << "[Rank 1] Posting an asynchronous receive (MPI_Irecv)..." << std::endl;

        // 1. 先掛起一個「非阻塞接收」，確保接收窗口已打開
        MPI_Irecv(recv_buf.data(), TEST_SIZE, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD, &request);

        // 2. 發送 0 長度的消息給 Rank 0，告訴它：「我準備好了！」
        std::cout << "[Rank 1] Signaling Rank 0 that I am ready." << std::endl;
        MPI_Send(nullptr, 0, MPI_INT, 0, TAG, MPI_COMM_WORLD);

        // 3. 等待真正的數據到達
        MPI_Wait(&request, &status);
        std::cout << "[Rank 1] Successfully received Rsend message from Rank " 
                  << status.MPI_SOURCE << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    test_rsend();
    MPI_Finalize();
    return 0;
}