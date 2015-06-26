package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class DenseSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  private final int[] outputDims;
  public final NDArray weights;

  private boolean frozen = false;
  
  public DenseSynapseLayer(int inputs, int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final NDArray output = new NDArray(outputDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = this.frozen?null:new NDArray(weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        double a = weights.get(i, o);
        double b = input.data[i];
        inputGradient.add(new int[] { i, o }, a);
        if(null!=weightGradient) weightGradient.add(new int[] { weights.index(i, o), o }, b);
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        if(null!=weightGradient) {
          ctx.adjust(DenseSynapseLayer.this, weights, ctx.invertFeedback(weightGradient, data.data));
        }
        if (inObj.isAlive()) {
          inObj.feedback(new NDArray(inputDims, ctx.invertFeedback(inputGradient, data.data)), ctx);
        }
      }
      
      public boolean isAlive() {
        return null!=weightGradient||inObj.isAlive();
      }
    };
  }

  public DenseSynapseLayer addWeights(DoubleSupplier f) {
    for(int i=0;i<weights.data.length;i++)
    {
      weights.data[i] = f.getAsDouble();
    }
    return this;
  }

  public DenseSynapseLayer setWeights(DoubleSupplier f) {
    Arrays.parallelSetAll(weights.data, i->f.getAsDouble());
    return this;
  }

  public DenseSynapseLayer freeze() {
    return freeze(true);
  }

  public DenseSynapseLayer thaw() {
    return freeze(false);
  }

  public DenseSynapseLayer freeze(boolean b) {
    this.frozen = b;
    return this;
  }
  
}
