package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;

public class PipelineNetwork extends NNLayer {
  private static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  protected List<NNLayer> layers = new ArrayList<NNLayer>();
  
  public PipelineNetwork add(final NNLayer layer) {
    this.layers.add(layer);
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult array) {
    NNResult r = array;
    for (final NNLayer l : this.layers) {
      r = l.eval(r);
    }
    return r;
  }
  
  protected BiasLayer mutate(final BiasLayer l, final double amount) {
    final Random random = new Random();
    double[] a = l.bias;
    for(int i=0;i<a.length;i++)
    {
      if(random.nextDouble() < amount) {
        a[i] = random.nextGaussian();
      }
    }
    return l;
  }
  
  protected DenseSynapseLayer mutate(final DenseSynapseLayer l, final double amount) {
    final Random random = new Random();
    double[] a = l.weights.data;
    for(int i=0;i<a.length;i++)
    {
      if(random.nextDouble() < amount) {
        a[i] = random.nextGaussian();
      }
    }
    return l;
  }
  
  protected PipelineNetwork mutate(final double amount) {
    this.layers.stream()
        .filter(l -> (l instanceof DenseSynapseLayer))
        .forEach(l -> mutate((DenseSynapseLayer) l, amount));
    this.layers.stream()
        .filter(l -> (l instanceof BiasLayer))
        .forEach(l -> mutate((BiasLayer) l, amount));
    return this;
  }
  
  void writeDeltas() {
    for(NNLayer l : layers) {
      if(l instanceof DeltaTransaction) ((DeltaTransaction)l).write();
    }
  }

  @Override
  public String toString() {
    return "PipelineNetwork [" + layers + "]";
  }

  public double train(final NDArray[][] samples, final int maxIterations, final double minRms) {
    return new Trainer().train(samples, maxIterations, minRms, this);
  }

  public PipelineNetwork test(final NDArray[][] samples, final int maxIterations, final double minRms, int loops) {
    return new Trainer().test(samples, maxIterations, minRms, loops, this);
  }
  
}