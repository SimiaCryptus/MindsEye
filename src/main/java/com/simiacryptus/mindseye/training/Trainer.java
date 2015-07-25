package com.simiacryptus.mindseye.training;

import groovy.lang.Tuple2;

import java.util.List;
import java.util.stream.IntStream;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;

public class Trainer {
  MacroTrainer macroTrainer = new MacroTrainer();

  public Trainer add(SupervisedTrainingParameters params) {
    macroTrainer.getInner().inner.current.add(params);
    return this;
  }

  public Trainer add(PipelineNetwork pipelineNetwork, NDArray[][] samples) {
    return add(new SupervisedTrainingParameters(pipelineNetwork, samples));
  }

  public Trainer setMutationAmount(double d) {
    macroTrainer.setMutationAmount(d);
    return this;
  }

  public Trainer setImprovementStaleThreshold(int i) {
    return this;
  }

  public Trainer setStaticRate(double d) {
    macroTrainer.getInner().setRate(d);
    return this;
  }

  public Trainer setVerbose(boolean b) {
    macroTrainer.setVerbose(true);
    return this;
  }

  public Trainer setRateAdaptionRate(double d) {
    macroTrainer.getInner().setRateAdaptionRate(d);
    return this;
  }

  public Trainer setDynamicRate(double d) {
    macroTrainer.getInner().setRate(d);
    return this;
  }

  public Trainer setMaxDynamicRate(double d) {
    macroTrainer.getInner().setMaxRate(d);
    return this;
  }

  public Trainer setMinDynamicRate(double d) {
    macroTrainer.getInner().setMinRate(d);
    return this;
  }

  public void train(int i, double d) {
    macroTrainer.setMaxIterations(i).setStopError(d);
    macroTrainer.train();
  }

  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    return new Tuple2<List<SupervisedTrainingParameters>, Double>(macroTrainer.getBest().currentNetworks, macroTrainer.getBest().error());
  }

  public void verifyConvergence(int maxIter, double convergence, int reps) {
    if(!IntStream.range(0, reps).allMatch(i->{
      return Util.kryo().copy(macroTrainer).setMaxIterations(maxIter).setStopError(convergence).train() <= convergence;
    })) throw new RuntimeException();
  }
  
}
