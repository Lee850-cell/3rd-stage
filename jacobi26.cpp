#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define SIZE 6              /* 數據陣列大小 */
#define MSG_TAG 2000        /* 統一的消息標籤 */

/* 全域變量：定義發送者與接收者 */
static int src = 0;
static int dest = 1;

/* 函數原型宣告 */
void Generate_Data(double *buffer, int buff_size);
void Normal_Test_Recv(double *buffer, int buff_size);
void Buffered_Test_Send(double *buffer, int buff_size);

int main(int argc, char **argv) {
    int rank, numprocs;
    double buffer[SIZE];
    double *tmpbuffer, *tmpbuf;
    int tsize, bsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    /* 檢查進程數：本程序嚴格要求 2 個進程 */
    if (numprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "*** This program requires exactly 2 processes! ***\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == src) {
        // --- 發送端邏輯 ---
        Generate_Data(buffer, SIZE);

        /* 1. 計算緩衝區所需大小 */
        MPI_Pack_size(SIZE, MPI_DOUBLE, MPI_COMM_WORLD, &bsize);

        /* 2. 申請記憶體：數據大小 + MPI 緩衝管理開銷 */
        tmpbuffer = (double *)malloc(bsize + MPI_BSEND_OVERHEAD);
        if (!tmpbuffer) {
            fprintf(stderr, "Error: Could not allocate bsend buffer of size %d\n", bsize);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* 3. 裝載緩衝區 (Attach) */
        MPI_Buffer_attach(tmpbuffer, bsize + MPI_BSEND_OVERHEAD);

        /* 4. 執行緩衝發送 */
        Buffered_Test_Send(buffer, SIZE);

        /* 5. 卸載並回收緩衝區 (Detach) */
        MPI_Buffer_detach(&tmpbuf, &tsize);
        free(tmpbuf); // 釋放 malloc 的記憶體

    } else if (rank == dest) {
        // --- 接收端邏輯 ---
        Normal_Test_Recv(buffer, SIZE);
    }

    MPI_Finalize();
    return 0;
}

/* --- 輔助函數實作 --- */

// 產生測試數據：1.0, 2.0, 3.0...
void Generate_Data(double *buffer, int buff_size) {
    for (int i = 0; i < buff_size; i++) {
        buffer[i] = (double)i + 1;
    }
}

// 標準接收函數
void Normal_Test_Recv(double *buffer, int buff_size) {
    MPI_Status stat;
    double *ptr = buffer;

    /* 分兩次接收：先接前 5 個，再接最後 1 個 */
    MPI_Recv(ptr, buff_size - 1, MPI_DOUBLE, src, MSG_TAG, MPI_COMM_WORLD, &stat);
    fprintf(stderr, "Standard Recv: Received %d elements.\n", buff_size - 1);
    for (int j = 0; j < buff_size - 1; j++) {
        fprintf(stderr, "  buf[%d] = %f\n", j, ptr[j]);
    }

    ptr += (buff_size - 1); // 指標後移
    MPI_Recv(ptr, 1, MPI_DOUBLE, src, MSG_TAG, MPI_COMM_WORLD, &stat);
    fprintf(stderr, "Standard Recv: Received last 1 element.\n");
    fprintf(stderr, "  buf[0] = %f\n", *ptr);
}

// 緩衝發送函數
void Buffered_Test_Send(double *buffer, int buff_size) {
    void *temp_ptr;
    int temp_size;
    double *ptr = buffer;

    fprintf(stderr, "Buffered Send: Sending %d elements...\n", buff_size - 1);
    MPI_Bsend(ptr, buff_size - 1, MPI_DOUBLE, dest, MSG_TAG, MPI_COMM_WORLD);

    ptr += (buff_size - 1);
    fprintf(stderr, "Buffered Send: Sending last 1 element...\n");
    MPI_Bsend(ptr, 1, MPI_DOUBLE, dest, MSG_TAG, MPI_COMM_WORLD);

    /* * 這裡的原碼中有一個 Detach/Attach 操作 
     * 目的是強制刷新緩衝區，確保消息送出
     */
    MPI_Buffer_detach(&temp_ptr, &temp_size);
    MPI_Buffer_attach(temp_ptr, temp_size);
}
