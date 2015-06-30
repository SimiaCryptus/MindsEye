package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.test.SimpleNetworkTests;

public class PipelineNetwork extends NNLayer {
  static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  private int improvementStaleThreshold = 20;
  private List<NNLayer> layers = new ArrayList<NNLayer>();
  private double mutationAmount = 0.1;
  private double rate = 0.00001;
  private boolean verbose = false;
  
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
  
  public double getMutationAmount() {
    return this.mutationAmount;
  }
  
  public double getRate() {
    return this.rate;
  }
  
  public double getRate(final int iteration) {
    return this.rate;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  protected BiasLayer mutate(final BiasLayer l, final double amount) {
    final Random random = new Random();
    l.addWeights(() -> amount * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
    return l;
  }
  
  protected DenseSynapseLayer mutate(final DenseSynapseLayer l, final double amount) {
    final Random random = new Random();
    l.addWeights(() -> amount * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
    return l;
  }
  
  protected PipelineNetwork mutate(final double amount) {
    if(verbose) log.debug(String.format("Mutating %s by %s", this, amount));
    this.layers.stream()
        .filter(l -> (l instanceof DenseSynapseLayer))
        .forEach(l -> mutate((DenseSynapseLayer) l, amount));
    this.layers.stream()
        .filter(l -> (l instanceof BiasLayer))
        .forEach(l -> mutate((BiasLayer) l, amount));
    return this;
  }
  
  public PipelineNetwork setMutationAmount(final double mutationAmount) {
    this.mutationAmount = mutationAmount;
    return this;
  }
  
  public PipelineNetwork setRate(final double rate) {
    this.rate = rate;
    return this;
  }
  
  public PipelineNetwork setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public void test(final NDArray[][] samples, final int maxIterations, final double convergence, final int trials) {
    final Kryo kryo = new Kryo();
    for (int epoch = 0; epoch < trials; epoch++)
    {
      // BUG: The previous network's state ensures future trials succeed immediately.
      final double rms = kryo.copy(this).mutate(1.).train(samples, maxIterations, convergence);
      if (isVerbose()) {
        log.info("Final RMS Error: {}", rms);
      }
      if (!Double.isFinite(rms) || rms >= convergence) throw new RuntimeException("Failed in trial " + epoch);
    }
  }
  
  protected double train(final NDArray[][] samples, final int maxIterations, final double convergence) {
    PipelineNetwork net = this;
    PipelineNetwork best = net;
    int timesSinceImprovement = 0;
    double bestRms = Double.MAX_VALUE;
    double rms = 0;
    
    for (final NDArray[] sample : samples) {
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      rms += net.eval(input).errRms(output);
    }
    rms /= samples.length;
    if (net.isVerbose()) {
      log.info("Starting RMS Error: {}", rms);
    }
    for (int i = 0; i < maxIterations; i++)
    {
      boolean mutated = false;
      boolean shouldMutate;
      if (bestRms <= rms)
        shouldMutate = timesSinceImprovement++ > this.improvementStaleThreshold;
      else
      {
        if(isVerbose()) {
          log.debug(String.format("New best RMS %s > %s", rms, bestRms));
        }
        best = new Kryo().copy(this);
        bestRms = rms;
        timesSinceImprovement = 0;
        shouldMutate = false;
      }
      if (shouldMutate) {
        //net = new Kryo().copy(net);
        mutated = true;
        net.mutate(net.getMutationAmount());
      }
      rms = 0;
      int count = 0;
      for (int rep = 0; rep < 1; rep++)
        for (final NDArray[] sample : samples) {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          final double rate = net.getRate(i);
          final NNResult eval = net.eval(input);
          final double trialRms = eval.errRms(output);
          rms += trialRms;
          count++;
          final NDArray delta = eval.delta(rate, output);
          eval.feedback(delta);
          if (net.isVerbose()) {
            // assert(net.eval(input).errRms(output) < trialRms) : "A marginal local improvement was expected";
          }
        }
      rms /= count;
      if (mutated) {
        double improvement = bestRms - rms;
        if (improvement <= 0) {
          if(verbose) log.debug("Discarding " + net);
          net = best;
          rms = bestRms;
          //log.debug("Restored rms: " + bestRms);
        }
      }
      if (rms < convergence) break;
      if (net.isVerbose()) {
        log.info(String.format("RMS Error: %s", rms));
      }
    }
    return rms;
  }

  @Override
  public String toString() {
    return "PipelineNetwork [" + layers + "]";
  }
  
}