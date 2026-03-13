#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    const int TOTAL_SIZE = 16;
    const int STEPS = 10;
    int myid, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 4 processes are required
    if (numprocs != 4) {
        if (myid == 0) std::cerr << "Are you fucking kidding me?" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int MY_SIZE = TOTAL_SIZE / numprocs;
    // Define a, b arrays (16 rows, MY_SIZE + 2 columns)
    std::vector<std::vector<double>> a(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));
    std::vector<std::vector<double>> b(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));

    // --- Initialization ---
    if (myid == 0) {
        for (int i = 0; i < TOTAL_SIZE; i++) a[i][1] = 8.0; // Left boundary
    }
    if (myid == 3) {
        for (int i = 0; i < TOTAL_SIZE; i++) a[i][MY_SIZE] = 8.0; // Right boundary
    }
    for (int j = 0; j < MY_SIZE + 2; j++) {
        a[0][j] = 8.0;             // Upper boundary
        a[TOTAL_SIZE - 1][j] = 8.0; // Lower boundary
    }

    // --- Jacobi iteration ---
    for (int n = 0; n < STEPS; n++) {
        MPI_Status status;
        std::vector<double> send_buf(TOTAL_SIZE);
        std::vector<double> recv_buf(TOTAL_SIZE);

        // 1. Shift data from left to right (exchange right boundary)
        if (myid == 0) {
            for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][MY_SIZE];
            MPI_Send(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10, MPI_COMM_WORLD);
        } else if (myid == 3) {
            MPI_Recv(recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10, MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][0] = recv_buf[i];
        } else {
            // use Sendrecv in the middle process: send data to right neighbor,
            // and receive data from left neighbor
            for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][MY_SIZE];
            MPI_Sendrecv(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10,
                         recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10,
                         MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][0] = recv_buf[i];
        }

        // 2. shift data from right to left (exchange left boundary)
        if (myid == 0) {
            MPI_Recv(recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10, MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][MY_SIZE + 1] = recv_buf[i];
        } else if (myid == 3) {
            for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][1];
            MPI_Send(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10, MPI_COMM_WORLD);
        } else {
            // use Sendrecv in the middle process: send data to left neighbor,
            // and receive data from right neighbor
            for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][1];
            MPI_Sendrecv(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid - 1, 10,
                         recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, myid + 1, 10,
                         MPI_COMM_WORLD, &status);
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][MY_SIZE + 1] = recv_buf[i];
        }

        // --- compute ---
        int begin_col = 1, end_col = MY_SIZE;
        if (myid == 0) begin_col = 2;
        if (myid == 3) end_col = MY_SIZE - 1;

        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) {
                b[i][j] = (a[i][j + 1] + a[i][j - 1] + a[i + 1][j] + a[i - 1][j]) * 0.25;
            }
        }
        // update A
        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) a[i][j] = b[i][j];
        }
    }

    // --- output ---
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