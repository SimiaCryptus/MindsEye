package com.simiacryptus.mindseye.net.util;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.ml.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

@SuppressWarnings("serial")
public final class WeightExtractor extends NNLayer<WeightExtractor> {

  static final Logger log = LoggerFactory.getLogger(WeightExtractor.class);

  private final NNLayer<?> inner;
  private final int index;

  public WeightExtractor(final int index, final NNLayer<?> inner) {
    this.inner = inner;
    this.index = index;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NDArray array = new NDArray(inner.state().get(index));
    return new NNResult(array) {
      
      @Override
      public boolean isAlive() {
        return true;
      }
      
      @Override
      public void accumulate(DeltaSet buffer, NDArray[] data) {
        assert(data.length==1);
        buffer.get(WeightExtractor.this, array).feed(data[0].getData());;
      }
    };
  }

  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
}
