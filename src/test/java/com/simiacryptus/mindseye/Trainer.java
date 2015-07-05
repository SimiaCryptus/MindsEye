package com.simiacryptus.mindseye;

import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.learning.NNResult;

public class Trainer {
  static final Logger log = LoggerFactory.getLogger(Trainer.class);
  private PipelineNetwork net;
  private NDArray[][] trainingData;
  private double mutationAmount = 0.5;
  int improvementStaleThreshold = 20;
  double dynamicRate = 0.1;
  double staticRate = 0.1;
  private boolean verbose = false;

  public Trainer(PipelineNetwork net, final NDArray[][] trainingData) {
    this.net = net;
    this.trainingData = trainingData;
  }

  public double train(final int maxIterations, final double minRms) {
    long startMs = System.currentTimeMillis();
    PipelineNetwork best = net;
    int timesSinceImprovement = 0;
    double bestRms = Double.MAX_VALUE;
    double rms = Double.MAX_VALUE;
    int totalIterations=0;
    int lessons = 5;
    int generations = maxIterations / lessons;
    for (int generation = 0; generation < generations; generation++)
    {
      boolean mutated = false;
      boolean shouldMutate;
      if (bestRms <= rms)
        shouldMutate = timesSinceImprovement++ > improvementStaleThreshold;
      else
      {
        if(isVerbose()) {
          log.debug(String.format("New best RMS %s > %s", rms, bestRms));
        }
        best = new Kryo().copy(net);
        bestRms = rms;
        timesSinceImprovement = 0;
        shouldMutate = false;
      }
      if (shouldMutate) {
        mutated = true;
        double mutationAmount = getMutationAmount();
        if(verbose) log.debug(String.format("Mutating %s by %s", this, mutationAmount));
        net.mutate(mutationAmount);
      }
      rms = 0;
      int count = 0;
      double lastRms = Double.NaN;
      for (int schoolDay = 0; schoolDay < lessons; schoolDay++) {
        double thisRms = 0;
        for (int lesson = 0; lesson < lessons; lesson++) {
          thisRms += trainSet();
        }
        thisRms /= lessons;
        if(Double.isFinite(lastRms)) {
          double expectedImprovement = lastRms * staticRate/(10+totalIterations);
          double improvement = lastRms - thisRms;
          if(0. == improvement) {
            if(isVerbose()) log.debug("Null improvement: " + net);
            if(isVerbose()) log.debug(String.format("Discarding %s rms: %s", rms, net));
            net = best;
            rms = bestRms;
            break;
          }
          double idealRate = dynamicRate * expectedImprovement / improvement;
          double critical = 10;
          if(idealRate > critical) idealRate = critical;
          else if(idealRate < -critical) idealRate = -critical;
          double prevRate = dynamicRate;
          if(isVerbose()) log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
          dynamicRate += 0.1 * (idealRate-dynamicRate);
          if(isVerbose()) log.debug(String.format("Rate %s -> %s", prevRate, dynamicRate));
        }
        totalIterations += trainingData.length;
        count += trainingData.length;
        rms += thisRms * lessons;
        lastRms = thisRms;
      }
      rms /= count;
      if (mutated) {
        double improvement = bestRms - rms;
        if (improvement <= 0) {
          if(isVerbose()) log.debug(String.format("Discarding %s rms: %s", rms, net));
          net = best;
          rms = bestRms;
        }
      }
      if (rms < minRms) break;
      if (isVerbose()) {
        log.info(String.format("RMS Error: %s", rms));
      }
    }
    log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", rms, (System.currentTimeMillis()-startMs)/1000., totalIterations));
    return rms;
  }

  public PipelineNetwork test(final int maxIterations, final double convergence, final int trials) {
    final Kryo kryo = new Kryo();
    PipelineNetwork lastGood = null;
    PipelineNetwork masterCopy = net;
    for (int epoch = 0; epoch < trials; epoch++)
    {
      net = kryo.copy(masterCopy).mutate(1.);
      final double rms = train(maxIterations, convergence);
      
      if (isVerbose()) {
        log.info("Final RMS Error: {}", rms);
      }
      if (!Double.isFinite(rms) || rms >= convergence) {
        log.info(String.format("Failed to converge in trial %s; Best RMS=%s: %s", epoch, rms, net));
        throw new RuntimeException("Failed in trial " + epoch);
      } else {
        lastGood = net;
      }
    }
    net = lastGood;
    return lastGood;
  }

  public double trainSet() {
    double rms = Stream.of(trainingData).parallel().mapToDouble(sample->{
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      final NNResult eval = net.eval(input);
      final double trialRms = eval.errRms(output);
      final NDArray delta = eval.delta(dynamicRate, output);
      eval.feedback(delta);
      return trialRms;
    }).sum();
    net.writeDeltas();
    if(verbose) log.debug(String.format("Trained Iteration RMS: %s with rate %s", rms, dynamicRate));
    return rms;
  }

  public double getMutationAmount() {
    return this.mutationAmount;
  }
  
  public Trainer setMutationAmount(final double mutationAmount) {
    this.mutationAmount = mutationAmount;
    return this;
  }
  
  public Trainer setRate(final double rate) {
    this.staticRate = rate;
    return this;
  }

  public boolean isVerbose() {
    return this.verbose;
  }
  
  public Trainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
