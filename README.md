# MPI Jacobi Iteration

這是一個使用 MPI 實現的 Jacobi 迭代並行程式，用於求解二維熱傳導問題。

## 編譯方法
```bash
mpiicpc -o jacobi jacobi.cpp -O3
執行方法
bash
mpirun -np 4 ./jacobi
檔案說明
jacobi.cpp - 主程式
