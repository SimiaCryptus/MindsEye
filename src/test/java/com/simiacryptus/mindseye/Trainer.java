package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.learning.NNResult;

public class Trainer {
  static final Logger log = LoggerFactory.getLogger(Trainer.class);
  private double mutationAmount = 0.5;
  private int improvementStaleThreshold = 20;
  double dynamicRate = 0.1;
  double staticRate = 1.;
  private boolean verbose = false;
  private List<SupervisedTrainingParameters> net = new ArrayList<>();
  
  public Trainer() {
  }

  public Trainer add(SupervisedTrainingParameters params) {
    this.net.add(params);
    return this;
  }

  public Trainer add(PipelineNetwork net, NDArray[][] data) {
    return add(new SupervisedTrainingParameters(net, data));
  }
  
  @SuppressWarnings("unchecked")
  public double train(final int maxIterations, final double minRms) {
    long startMs = System.currentTimeMillis();
    Object best = net;
    int timesSinceImprovement = 0;
    double bestRms = Double.MAX_VALUE;
    double rms = Double.MAX_VALUE;
    int totalIterations = 0;
    int lessons = 5;
    int generations = maxIterations / lessons;
    for (int generation = 0; generation < generations; generation++)
    {
      boolean mutated = false;
      boolean shouldMutate;
      if (!Double.isFinite(rms) || bestRms <= rms)
        shouldMutate = timesSinceImprovement++ > improvementStaleThreshold;
      else
      {
        if (isVerbose()) {
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
        if (verbose) log.debug(String.format("Mutating %s by %s", net, mutationAmount));
        net.stream().forEach(x -> x.getNet().mutate(mutationAmount));
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
        if (Double.isFinite(lastRms)) {
          double expectedImprovement = lastRms * staticRate / (50 + totalIterations);
          double improvement = lastRms - thisRms;
          if (0. == improvement) {
            if (isVerbose()) log.debug("Null improvement: " + net);
            if (isVerbose()) log.debug(String.format("Discarding %s rms: %s", rms, net));
            net = (List<SupervisedTrainingParameters>) best;
            rms = 0;
            count = 0;
            break;
          }
          double idealRate = dynamicRate * expectedImprovement / improvement;
          double prevRate = dynamicRate;
          if (isVerbose())
            log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
          dynamicRate += Math.max(Math.min(0.1 * (idealRate - dynamicRate), 1.), -1);
          if (isVerbose()) log.debug(String.format("Rate %s -> %s", prevRate, dynamicRate));
        }
        totalIterations += 1;
        count += lessons;
        rms += thisRms * lessons;
        lastRms = thisRms;
      }
      rms /= count;
      if (mutated) {
        double improvement = bestRms - rms;
        if (improvement <= 0) {
          if (isVerbose()) log.debug(String.format("Discarding %s rms, best = %s", rms, bestRms));
          net = (List<SupervisedTrainingParameters>) best;
          rms = bestRms;
        }
      }
      if (rms < minRms) break;
      if (isVerbose()) {
        log.info(String.format("RMS Error: %s", rms));
      }
    }
    log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", rms, (System.currentTimeMillis() - startMs) / 1000., totalIterations));
    return rms;
  }
  
  public void test(final int maxIterations, final double convergence, final int trials) {
    final Kryo kryo = new Kryo();
    List<SupervisedTrainingParameters> lastGood = null;
    List<SupervisedTrainingParameters> masterCopy = net;
    for (int epoch = 0; epoch < trials; epoch++)
    {
      net = kryo.copy(masterCopy);
      net.stream().forEach(x -> x.getNet().mutate(1.));
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
  }
  
  public double trainSet() {
    double rms = net.stream().flatMapToDouble(params -> Stream.of(params.getTrainingData()).parallel().mapToDouble(sample -> {
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      final NNResult eval = params.getNet().eval(input);
      final double trialRms = eval.errRms(output);
      final NDArray delta = eval.delta(dynamicRate, output);
      eval.feedback(delta);
      return trialRms;
    })).average().getAsDouble();
    net.stream().forEach(params -> params.getNet().writeDeltas());
    if (verbose) log.debug(String.format("Trained Iteration RMS: %s with rate %s", rms, dynamicRate));
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

  public int getImprovementStaleThreshold() {
    return improvementStaleThreshold;
  }

  public Trainer setImprovementStaleThreshold(int improvementStaleThreshold) {
    this.improvementStaleThreshold = improvementStaleThreshold;
    return this;
  }
}
