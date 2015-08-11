package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.MassParameters;
import com.simiacryptus.mindseye.learning.NNResult;

public class PipelineNetwork extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  public static final Random random = new Random(System.nanoTime());
  protected List<NNLayer> layers = new ArrayList<NNLayer>();
  private double mutationAmplitude = 2.;
  
  public PipelineNetwork add(final NNLayer layer) {
    this.layers.add(layer);
    return this;
  }
  
  public PipelineNetwork clearMomentum() {
    this.layers.stream().filter(a -> a instanceof MassParameters<?>).map(a -> (MassParameters<?>) a).forEach(a -> a.setMomentumDecay(0));
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

  public NNLayer get(final int i) {
    return this.layers.get(i);
  }
  
  public double getMutationAmplitude() {
    return this.mutationAmplitude;
  }
  
  protected BiasLayer mutate(final BiasLayer l, final double amount) {
    final double[] a = l.bias;
    for (int i = 0; i < a.length; i++)
    {
      if (PipelineNetwork.random.nextDouble() < amount) {
        a[i] = this.mutationAmplitude * l.getRandom();
      }
    }
    return l;
  }

  protected DenseSynapseLayer mutate(final DenseSynapseLayer l, final double amount) {
    final double[] a = l.weights.getData();
    for (int i = 0; i < a.length; i++)
    {
      if (PipelineNetwork.random.nextDouble() < amount) {
        a[i] = this.mutationAmplitude * l.getRandom();
      }
    }
    return l;
  }

  protected PipelineNetwork mutate(final double amount) {
    this.layers.stream()
    .filter(l -> (l instanceof DenseSynapseLayer))
    .map(l -> (DenseSynapseLayer) l)
    .filter(l -> !l.isFrozen())
    .forEach(l -> mutate(l, amount));
    this.layers.stream()
    .filter(l -> (l instanceof BiasLayer))
    .map(l -> (BiasLayer) l)
    .filter(l -> !l.isFrozen())
    .forEach(l -> mutate(l, amount));
    return this;
  }

  public PipelineNetwork setMutationAmplitude(final double mutationAmplitude) {
    this.mutationAmplitude = mutationAmplitude;
    return this;
  }
  
  @Override
  public String toString() {
    return "PipelineNetwork [" + this.layers + "]";
  }
  
  public Trainer trainer(final NDArray[][] samples) {
    return new Trainer().add(this, samples);
  }
  
  void writeDeltas(final double factor) {
    this.layers.stream() //
      .filter(x->x instanceof DeltaTransaction) //
      .map(x->(DeltaTransaction)x) //
      .forEach(x->x.write(factor));
  }
  
}