#!/usr/bin/env python3
"""
Test nonzero behavior in MindSpore
"""

import mindspore as ms
import mindspore.ops as ops

# Test different cases
print("Testing ops.nonzero behavior...")

# Case 1: All False
tensor1 = ms.Tensor([False, False, False])
result1 = ops.nonzero(tensor1)
print(f"All False - Result type: {type(result1)}, Length: {len(result1) if hasattr(result1, '__len__') else 'N/A'}")
print(f"Result: {result1}")

# Case 2: Some True
tensor2 = ms.Tensor([True, False, True])
result2 = ops.nonzero(tensor2)
print(f"\nSome True - Result type: {type(result2)}, Length: {len(result2) if hasattr(result2, '__len__') else 'N/A'}")
print(f"Result: {result2}")

# Case 3: All True
tensor3 = ms.Tensor([True, True, True])
result3 = ops.nonzero(tensor3)
print(f"\nAll True - Result type: {type(result3)}, Length: {len(result3) if hasattr(result3, '__len__') else 'N/A'}")
print(f"Result: {result3}")

# Test proper way to check for empty
print("\n\nTesting proper empty check:")
for tensor, name in [(tensor1, "All False"), (tensor2, "Some True"), (tensor3, "All True")]:
    result = ops.nonzero(tensor)
    print(f"{name}:")
    try:
        if isinstance(result, tuple):
            if len(result) > 0 and result[0].size > 0:
                print(f"  Has elements: {result[0]}")
            else:
                print(f"  Empty result")
        else:
            if result.size > 0:
                print(f"  Has elements: {result}")
            else:
                print(f"  Empty result")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Result is: {result}")