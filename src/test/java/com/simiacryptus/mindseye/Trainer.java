package com.simiacryptus.mindseye;

import groovy.lang.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.simiacryptus.mindseye.learning.NNResult;

public class Trainer {
  static final Logger log = LoggerFactory.getLogger(Trainer.class);
  private double mutationAmount = 0.2;
  private int improvementStaleThreshold = 10;
  double dynamicRate = 0.1;
  double staticRate = 1.;
  private boolean verbose = false;
  private List<SupervisedTrainingParameters> net = new ArrayList<>();
  private Tuple2<List<SupervisedTrainingParameters>, Double> best = null;
  int timesSinceImprovement = 0;
  
  public Trainer() {
  }
  
  public Trainer add(SupervisedTrainingParameters params) {
    this.net.add(params);
    return this;
  }
  
  public Trainer add(PipelineNetwork net, NDArray[][] data) {
    return add(new SupervisedTrainingParameters(net, data));
  }
  
  public double train(final int maxIterations, final double minRms) {
    long startMs = System.currentTimeMillis();
    int totalIterations = 0;
    int lessons = 5;
    int generations = maxIterations / lessons;
    for (int generation = 0; generation < generations; generation++)
    {
      maybeMutate();
      double rms = update(totalIterations, lessons);
      totalIterations += lessons * lessons;
      if (rms < minRms) {
        log.info(String.format("Completed training to %.5f in %.03fs (%s iterations)", rms, (System.currentTimeMillis() - startMs) / 1000., totalIterations));
        return rms;
      }
      if (isVerbose()) {
        log.info(String.format("RMS Error: %s", rms));
      }
    }
    return best.getSecond();
  }
  
  public double update(int totalIterations, int lessons) {
    double rms;
    rms = 0;
    int count = 0;
    double lastRms = Double.NaN;
    for (int schoolDay = 0; schoolDay < lessons; schoolDay++) {
      double thisRms = 0;
      for (int lesson = 0; lesson < lessons; lesson++) {
        double[] rms1 = trainSet();
        if (verbose)
          log.debug(String.format("Trained Iteration %s RMS: %s with rate %s: %n%s", totalIterations++, Arrays.toString(rms1), dynamicRate, net.get(0).getNet()));
        double lessonRms = DoubleStream.of(rms1).average().getAsDouble();
        updateBest(lessonRms);
        
        if (Double.isFinite(lastRms)) {
          if (0. == lastRms - lessonRms) {
            mutate();
            net = best.getFirst();
            thisRms = best.getSecond();
            timesSinceImprovement = 0;
          } else {
            updateRate(lastRms, lessonRms);
          }
        }
        lastRms = lessonRms;
        thisRms += lessonRms;
      }
      thisRms /= lessons;
      if (null != best && (timesSinceImprovement > improvementStaleThreshold * 2 || thisRms > 2. * best.getSecond())) {
        if (best.getSecond() <= thisRms) {
          if (isVerbose()) log.debug(String.format("Discarding %s rms, best = %s", thisRms, best.getSecond()));
          net = best.getFirst();
          thisRms = best.getSecond();
          timesSinceImprovement = 0;
        }
      }
      count += lessons;
      rms += thisRms * lessons;
    }
    rms /= count;
    return rms;
  }
  
  public void updateRate(double lastRms, double thisRms) {
    double improvement = lastRms - thisRms;
    double expectedImprovement = lastRms * staticRate / 50;// (50 + totalIterations);
    double idealRate = dynamicRate * expectedImprovement / improvement;
    double prevRate = dynamicRate;
    if (isVerbose())
      log.debug(String.format("Ideal Rate: %s (target %s change, actual %s with %s rate)", idealRate, expectedImprovement, improvement, prevRate));
    dynamicRate += 0.1 * (Math.max(Math.min(idealRate, 1.), -1) - dynamicRate);
    if (isVerbose()) log.debug(String.format("Rate %s -> %s", prevRate, dynamicRate));
  }
  
  int timeSinceMutation = 0;
  
  public void maybeMutate() {
    if (timesSinceImprovement - Math.min(timeSinceMutation, timesSinceImprovement) > improvementStaleThreshold) {
      mutate();
    }
  }
  
  public void mutate() {
    mutate(getMutationAmount());
  }
  
  public void mutate(double mutationAmount) {
    timeSinceMutation = timesSinceImprovement;
    if (verbose) log.debug(String.format("Mutating %s by %s", net, mutationAmount));
    net.stream().forEach(x -> x.getNet().mutate(mutationAmount));
  }
  
  public void updateBest(double rms) {
    if (!Double.isFinite(rms) || (null != best && best.getSecond() <= rms)) {
      timesSinceImprovement++;
    } else {
      if (isVerbose()) {
        log.debug(String.format("New best RMS %s > %s", rms, null == best ? "null" : best.getSecond()));
      }
      best = new Tuple2<List<SupervisedTrainingParameters>, Double>(new Kryo().copy(net), rms);
      timesSinceImprovement = 0;
    }
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
  
  public double[] trainSet() {
    double[] rms = net.stream().mapToDouble(params -> Stream.of(params.getTrainingData()).parallel().mapToDouble(sample -> {
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      final NNResult eval = params.getNet().eval(input);
      final double trialRms = eval.errRms(output);
      final NDArray delta = eval.delta(dynamicRate * params.getWeight(), output);
      eval.feedback(delta);
      assert (Double.isFinite(trialRms));
      return trialRms;
    }).average().getAsDouble()).toArray();
    net.stream().forEach(params -> params.getNet().writeDeltas());
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
  
  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    return best;
  }
  
  public void setBest(Tuple2<List<SupervisedTrainingParameters>, Double> best) {
    this.best = best;
  }
}
