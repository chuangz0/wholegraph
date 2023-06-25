# WholeMemory Implementation Details
As described in [WholeMemory Introduction](wholegraph_intro.md), there are two WholeMemory location and three
WholeMemory types. So there will be total six WholeMemory.

|     Type      | CONTINUOUS  | CONTINUOUS |  CHUNKED  |  CHUNKED  | DISTRIBUTED | DISTRIBUTED |
|:-------------:|:-----------:|:----------:|:---------:|:---------:|:-----------:|:-----------:|
|   Location    |   DEVICE    |    HOST    |  DEVICE   |   HOST    |   DEVICE    |    HOST     |
| Allocated by  |    EACH     |   FIRST    |   EACH    |   FIRST   |    EACH     |    EACH     |
| Allocate API  |   Driver    |    Host    |  Runtime  |   Host    |   Runtime   |   Runtime   |
|  IPC Mapping  |   Unix fd   |    mmap    |  cudaIpc  |   mmap    | No IPC map  | No IPC map  |
