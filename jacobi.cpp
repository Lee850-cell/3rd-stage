#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

const int TOTAL_SIZE = 16;      // 全局數組大小
const int MY_SIZE = TOTAL_SIZE / 4;  // 每個進程處理的大小
const int STEPS = 10;            // 迭代次數

int main(int argc, char** argv) {
    int myid, numprocs;
    
    MPI_Init(&argc, &argv);//initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);//get rank of current process
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);//get number of processes
    
    if (numprocs != 4) {
        if (myid == 0) {
            std::cerr << "Are you fucking serious?" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    

    std::cout << "Process " << myid << " of " << numprocs << " is alive" << std::endl;
    
    // initialize arrays
    std::vector<std::vector<double>> a(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));
    std::vector<std::vector<double>> b(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));
    
    // initialize boundary values
    if (myid == 0) {
        for (int i = 0; i < TOTAL_SIZE; i++) {
            a[i][1] = 8.0;  // 第2列（索引1）
        }
    }
    if (myid == 3) {
        for (int i = 0; i < TOTAL_SIZE; i++) {
            a[i][MY_SIZE] = 8.0;  // 倒數第2列
        }
    }
    
    // initialize boundary values
    for (int i = 0; i < MY_SIZE + 2; i++) {
        a[0][i] = 8.0;               // 第一行
        a[TOTAL_SIZE - 1][i] = 8.0;  // 最後一行
    }
    
    // Jacobi alternative
    for (int n = 0; n < STEPS; n++) {
        MPI_Status status;
        
        // recv from right neighbor
        if (myid < 3) {
            std::vector<double> recv_buf(TOTAL_SIZE);
            MPI_Recv(recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10, 
                     MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) {
                a[i][MY_SIZE + 1] = recv_buf[i];
            }
        }
        
        // send to left neighbor
        if (myid > 0) {
            std::vector<double> send_buf(TOTAL_SIZE);
            for (int i = 0; i < TOTAL_SIZE; i++) {
                send_buf[i] = a[i][1];
            }
            MPI_Send(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10, 
                     MPI_COMM_WORLD);
        }
        
        // send to right neighbor
        if (myid < 3) {
            std::vector<double> send_buf(TOTAL_SIZE);
            for (int i = 0; i < TOTAL_SIZE; i++) {
                send_buf[i] = a[i][MY_SIZE];
            }
            MPI_Send(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10, 
                     MPI_COMM_WORLD);
        }
        
        //  recv from left neighbor
        if (myid > 0) {
            std::vector<double> recv_buf(TOTAL_SIZE);
            MPI_Recv(recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10, 
                     MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) {
                a[i][0] = recv_buf[i];
            }
        }
        
        // calculate new values
        int begin_col = 1;      // 第2列（索引1）
        int end_col = MY_SIZE;  // 倒數第2列
        
        if (myid == 0) {
            begin_col = 2;  // except left boundary
        }
        if (myid == 3) {
            end_col = MY_SIZE - 1;  // except for right boundary
        }
        
        // compute new values
        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) {
                b[i][j] = (a[i][j+1] + a[i][j-1] + a[i+1][j] + a[i-1][j]) * 0.25;
            }
        }
        
        // update arrays
        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) {
                a[i][j] = b[i][j];
            }
        }
    }
    
    // output
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 1; i < TOTAL_SIZE - 1; i++) {
        std::cout << "Process " << myid << ": ";
        for (int j = 1; j <= MY_SIZE; j++) {
            std::cout << std::setw(8) << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}