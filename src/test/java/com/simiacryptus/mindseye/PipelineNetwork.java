package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.test.TestNetworkUnit;

public class PipelineNetwork extends NNLayer {
  static final Logger log = LoggerFactory.getLogger(TestNetworkUnit.class);
  
  private int improvementStaleThreshold = 20;
  private double lastRms = Double.MAX_VALUE;
  private List<NNLayer> layers = new ArrayList<NNLayer>();
  private double rate = 0.00001;
  private int timesSinceImprovement = 0;
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
  
  public double getRate() {
    return this.rate;
  }

  public double getRate(final int iteration) {
    final double exp = Math.exp(Math.random() * Math.random() * 4) / 2;
    return this.rate * exp;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  protected void mutate() {
    this.layers.stream()
        .filter(l -> (l instanceof DenseSynapseLayer))
        .forEach(l -> mutate((DenseSynapseLayer) l));
  }

  protected DenseSynapseLayer mutate(final DenseSynapseLayer l) {
    final Random random = new Random();
    l.addWeights(() -> 0.1 * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
    return l;
  }

  public PipelineNetwork setRate(final double rate) {
    this.rate = rate;
    return this;
  }

  public PipelineNetwork setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  protected boolean shouldMutate(final int i, final double rms) {
    final boolean improved = this.lastRms * 1. < rms;
    this.lastRms = rms;
    if (improved)
      return this.timesSinceImprovement++ > this.improvementStaleThreshold;
      else
      {
        this.timesSinceImprovement = 0;
        return false;
      }
  }
  
  public void test(final NDArray[][] samples, final int maxIterations, final double convergence, final int trials) {
    for (int epoch = 0; epoch < trials; epoch++)
    {
      // BUG: The previous network's state ensures future trials succeed immediately.
      final double rms = train(samples, maxIterations, convergence);
      PipelineNetwork.log.info("RMS Error: {}", rms);
      if (rms >= convergence) throw new RuntimeException("Failed in trial " + epoch);
    }
  }
  
  protected double train(final NDArray[][] samples, final int maxIterations, final double convergence) {
    final PipelineNetwork net = this;
    double rms = 0;
    for (final NDArray[] sample : samples) {
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      rms += net.eval(input).errRms(output);
    }
    rms /= samples.length;
    PipelineNetwork.log.info("RMS Error: {}", rms);
    for (int i = 0; i < maxIterations; i++)
    {
      if (shouldMutate(i, rms)) {
        mutate();
      }
      rms = 0;
      for (final NDArray[] sample : samples) {
        final NDArray input = sample[0];
        final NDArray output = sample[1];
        final double rate = getRate(i);
        final NNResult eval = net.eval(input);
        rms += eval.errRms(output);
        final NDArray delta = eval.delta(rate, output);
        eval.feedback(delta);
      }
      rms /= samples.length;
      if (rms < convergence) {
        break;
      }
      if (isVerbose()) {
        PipelineNetwork.log.info("RMS Error: {}", rms);
      }
    }
    return rms;
  }

}