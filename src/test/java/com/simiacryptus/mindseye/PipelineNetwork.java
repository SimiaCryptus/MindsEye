package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;

public class PipelineNetwork extends NNLayer {
  static final Logger log = LoggerFactory.getLogger(PipelineNetwork.class);
  
  private int improvementStaleThreshold = 20;
  protected List<NNLayer> layers = new ArrayList<NNLayer>();
  private double mutationAmount = 0.5;
  private double rate = 0.1;
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
  
  public boolean isVerbose() {
    return this.verbose;
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
  
  public PipelineNetwork test(final NDArray[][] samples, final int maxIterations, final double convergence, final int trials) {
    final Kryo kryo = new Kryo();
    PipelineNetwork lastGood = null;
    for (int epoch = 0; epoch < trials; epoch++)
    {
      PipelineNetwork student = kryo.copy(this).mutate(1.);
      final double rms = student.train(samples, maxIterations, convergence);
      if (isVerbose()) {
        log.info("Final RMS Error: {}", rms);
      }
      if (!Double.isFinite(rms) || rms >= convergence) {
        log.info(String.format("Failed to converge in trial %s; Best RMS=%s: %s", epoch, rms, student));
        throw new RuntimeException("Failed in trial " + epoch);
      } else {
        lastGood = student;
      }
    }
    return lastGood;
  }
  
  public double train(final NDArray[][] samples, final int maxIterations, final double minRms) {
    long startMs = System.currentTimeMillis();
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
    int totalIterations=0;
    int lessons = 5;
    int generations = maxIterations / lessons;
    for (int generation = 0; generation < generations; generation++)
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
        mutated = true;
        net.mutate(net.getMutationAmount());
      }
      rms = 0;
      int count = 0;
      double lastRms = Double.NaN;
      double rate = net.rate;
      for (int schoolDay = 0; schoolDay < lessons; schoolDay++) {
        double thisRms = 0;
        for (int lesson = 0; lesson < lessons; lesson++) {
          thisRms += net.trainSet(samples, rate);
        }
        thisRms /= lessons;
        if(Double.isFinite(lastRms)) {
          double expectedImprovement = lastRms * net.rate/(10+totalIterations);
          double improvement = lastRms - thisRms;
          if(0. == improvement) {
            if(verbose) log.debug("Null improvement: " + net);
            if(verbose) log.debug(String.format("Discarding %s rms: %s", rms, net));
            net = best;
            rms = bestRms;
            break;
          }
          double idealRate = rate * expectedImprovement / improvement;
          double critical = 10;
          if(idealRate > critical) idealRate = critical;
          else if(idealRate < -critical) idealRate = -critical;
          double prevRate = rate;
          if(verbose) log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
          rate += 0.1 * (idealRate-rate);
          if(verbose) log.debug(String.format("Rate %s -> %s", prevRate, rate));
        }
        totalIterations += samples.length;
        count += samples.length;
        rms += thisRms * lessons;
        lastRms = thisRms;
      }
      rms /= count;
      if (mutated) {
        double improvement = bestRms - rms;
        if (improvement <= 0) {
          if(verbose) log.debug(String.format("Discarding %s rms: %s", rms, net));
          net = best;
          rms = bestRms;
        }
      }
      if (rms < minRms) break;
      if (net.isVerbose()) {
        log.info(String.format("RMS Error: %s", rms));
      }
    }
    log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", rms, (System.currentTimeMillis()-startMs)/1000., totalIterations));
    return rms;
  }

  private void writeDeltas() {
    for(NNLayer l : layers) {
      if(l instanceof DeltaTransaction) ((DeltaTransaction)l).write();
    }
  }

  private double trainSet(final NDArray[][] samples, final double rate) {
    setRate(rate);
    double rms = Stream.of(samples).parallel().mapToDouble(sample->{
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      final NNResult eval = this.eval(input);
      final double trialRms = eval.errRms(output);
      final NDArray delta = eval.delta(rate, output);
      eval.feedback(delta);
      return trialRms;
    }).sum();
    writeDeltas();
    return rms;
  }

  @Override
  public String toString() {
    return "PipelineNetwork [" + layers + "]";
  }
  
}