package com.simiacryptus.mindseye.net.util;

import java.util.List;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

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
    Tensor array = new Tensor(inner.state().get(index));
    return new NNResult(array) {
      
      @Override
      public boolean isAlive() {
        return true;
      }
      
      @Override
      public void accumulate(DeltaSet buffer, Tensor[] data) {
        assert(data.length==1);
        buffer.get(WeightExtractor.this, array).accumulate(data[0].getData());;
      }
    };
  }

  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
}
