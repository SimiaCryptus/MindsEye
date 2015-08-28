package com.simiacryptus.mindseye.learning;

import java.util.LinkedHashMap;
import java.util.Map;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaBuffer {
  public final Map<NNLayer, DeltaFlushBuffer> map = new LinkedHashMap<>();
  
  public DeltaFlushBuffer get(NNLayer layer, double[] ptr){
    return map.computeIfAbsent(layer, l->new DeltaFlushBuffer(ptr, layer));
  }

  public DeltaFlushBuffer get(NNLayer layer, NDArray ptr) {
    return get(layer, ptr.getData());
  }
}
