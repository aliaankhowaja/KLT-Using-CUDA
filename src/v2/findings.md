## Findings/Results

We tested our program against three different datasets and this file mentions the whole program execution time(data transfers + actual computation) for each example and each dataset, both on GPU and CPU

**Dataset d0 (9 frames, 76 KB per frame)**

**Original Timings:**

| Iteration   | CPU (Example 1) | CPU (Example 2) | CPU (Example 3) | CPU (Example 4) | CPU (Example 5) | GPU (Example 1) | GPU (Example 2) | GPU (Example 3) | GPU (Example 4) | GPU (Example 5) |
| ----------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1           | 0.052           | 0.073           | 0.190           | 0.001           | 0.058           | 0.278           | 0.256           | 0.330           | 0.001           | 0.255           |
| 2           | 0.053           | 0.073           | 0.189           | 0.001           | 0.059           | 0.249           | 0.257           | 0.330           | 0.001           | 0.244           |
| 3           | 0.053           | 0.073           | 0.195           | 0.001           | 0.059           | 0.237           | 0.268           | 0.316           | 0.001           | 0.240           |
| 4           | 0.053           | 0.076           | 0.190           | 0.001           | 0.060           | 0.238           | 0.252           | 0.319           | 0.001           | 0.240           |
| 5           | 0.053           | 0.073           | 0.190           | 0.001           | 0.059           | 0.238           | 0.263           | 0.325           | 0.001           | 0.240           |
| 6           | 0.053           | 0.072           | 0.189           | 0.001           | 0.059           | 0.239           | 0.268           | 0.322           | 0.001           | 0.253           |
| 7           | 0.053           | 0.072           | 0.189           | 0.001           | 0.059           | 0.228           | 0.257           | 0.317           | 0.001           | 0.226           |
| 8           | 0.053           | 0.073           | 0.189           | 0.001           | 0.059           | 0.190           | 0.242           | 0.301           | 0.001           | 0.195           |
| 9           | 0.053           | 0.072           | 0.188           | 0.001           | 0.059           | 0.190           | 0.195           | 0.268           | 0.001           | 0.193           |
| 10          | 0.053           | 0.072           | 0.190           | 0.001           | 0.059           | 0.192           | 0.204           | 0.293           | 0.001           | 0.243           |
| **Average** | **0.054s**      | **0.073s**      | **0.190s**      | **0.001s**      | **0.059s**      | **0.244s**      | **0.257s**      | **0.317s**      | **0.001s**      | **0.240s**      |


For dataset d0, the GPU performs slower than the CPU due to overhead. The execution time for GPU is much higher compared to CPU. The GPU overhead outweighs the parallelization benefits, resulting in slower execution.

---

**Dataset d1 (270 frames, 396 KB per frame)**

**Original Timings:**

| Iteration   | CPU (Example 1) | CPU (Example 2) | CPU (Example 3) | CPU (Example 4) | CPU (Example 5) | GPU (Example 1) | GPU (Example 2) | GPU (Example 3) | GPU (Example 4) | GPU (Example 5) |
| ----------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1           | 0.274           | 0.397           | 18.132          | 0.028           | 0.309           | 0.462           | 0.460           | 7.835           | 0.030           | 0.410           |
| 2           | 0.273           | 0.391           | 18.034          | 0.028           | 0.308           | 0.380           | 0.481           | 7.854           | 0.029           | 0.425           |
| 3           | 0.269           | 0.389           | 17.946          | 0.028           | 0.305           | 0.383           | 0.458           | 7.930           | 0.029           | 0.415           |
| 4           | 0.271           | 0.386           | 17.879          | 0.028           | 0.305           | 0.380           | 0.470           | 7.840           | 0.029           | 0.411           |
| 5           | 0.270           | 0.388           | 18.015          | 0.028           | 0.306           | 0.370           | 0.487           | 8.540           | 0.030           | 0.401           |
| **Average** | **0.271s**      | **0.390s**      | **17.999s**     | **0.028s**      | **0.306s**      | **0.375s**      | **0.465s**      | **7.846s**      | **0.029s**      | **0.414s**      |


For dataset d1, the GPU significantly outperforms the CPU. The GPU has a noticeable speedup of approximately 2.3x compared to CPU, benefiting from parallelization with the larger dataset.

---

**Dataset d2 (250 frames, 8 MB per frame)**

**Original Timings:**

| Iteration   | CPU (Example 1) | CPU (Example 2) | CPU (Example 3) | CPU (Example 4) | CPU (Example 5) | GPU (Example 1) | GPU (Example 2) | GPU (Example 3) | GPU (Example 4) | GPU (Example 5) |
| ----------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1           | 6.584           | 9.675           | 438.680         | 0.028           | 7.418           | 3.845           | 6.074           | 141.513         | 0.029           | 4.574           |
| **Average** | **6.584s**      | **9.675s**      | **438.680s**    | **0.028s**      | **7.418s**      | **3.845s**      | **6.074s**      | **141.513s**    | **0.029s**      | **4.574s**      |


For dataset d2, the GPU performs significantly better, showing a speedup of approximately 3x. The larger dataset size allows the GPU to leverage its parallel processing capabilities more effectively.

---

**Conclusion:** On larger datasets, notable improvements are observed when KLT feature tracking is performed on the GPU, whereas, for smaller datasets, the performance actually slows down so CPU is more suitable for smaller datasets.