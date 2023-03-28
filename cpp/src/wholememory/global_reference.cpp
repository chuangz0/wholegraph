#include <wholememory/global_reference.h>

#ifdef __cplusplus
extern "C" {
#endif

wholememory_gref_t wholememory_create_continuous_global_reference(void* ptr)
{
  wholememory_gref_t gref;
  gref.stride  = 0;
  gref.pointer = ptr;
  return gref;
}

#ifdef __cplusplus
}
#endif