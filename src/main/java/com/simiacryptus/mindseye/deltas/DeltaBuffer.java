package com.simiacryptus.mindseye.deltas;

import java.util.LinkedHashMap;
import java.util.Map;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaBuffer {
  public final Map<NNLayer, DeltaFlushBuffer> map = new LinkedHashMap<>();

  public synchronized DeltaFlushBuffer get(final NNLayer layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaFlushBuffer(ptr, layer));
  }
  
  public DeltaFlushBuffer get(final NNLayer layer, final NDArray ptr) {
    return get(layer, ptr.getData());
  }
}
