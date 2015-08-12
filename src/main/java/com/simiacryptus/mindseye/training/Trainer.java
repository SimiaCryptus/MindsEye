package com.simiacryptus.mindseye.training;

import groovy.lang.Tuple2;

import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;

public class Trainer {
  static final Logger log = LoggerFactory.getLogger(Trainer.class);
  
  MacroTrainer macroTrainer = new MacroTrainer();
  
  public Trainer add(final PipelineNetwork pipelineNetwork, final NDArray[][] samples) {
    return add(new SupervisedTrainingParameters(pipelineNetwork, samples));
  }
  
  public Trainer add(final SupervisedTrainingParameters params) {
    this.macroTrainer.getInner().inner.current.add(params);
    return this;
  }
  
  public Tuple2<List<SupervisedTrainingParameters>, Double> getBest() {
    return new Tuple2<List<SupervisedTrainingParameters>, Double>(this.macroTrainer.getBest().currentNetworks, this.macroTrainer.getBest().error());
  }
  
  public Trainer setDynamicRate(final double d) {
    this.macroTrainer.getInner().inner.current.setRate(d);
    return this;
  }
  
  public Trainer setMaxDynamicRate(final double d) {
    this.macroTrainer.getInner().setMaxRate(d);
    return this;
  }
  
  public Trainer setMinDynamicRate(final double d) {
    this.macroTrainer.getInner().setMinRate(d);
    return this;
  }
  
  public Trainer setMutationAmount(final double d) {
    this.macroTrainer.setMutationAmount(d);
    return this;
  }
  
  public Trainer setStaticRate(final double d) {
    this.macroTrainer.getInner().setRate(d);
    return this;
  }
  
  public Trainer setVerbose(final boolean b) {
    this.macroTrainer.setVerbose(b);
    return this;
  }
  
  public void train(final int i, final double d) {
    this.macroTrainer.setMaxIterations(i).setStopError(d);
    this.macroTrainer.train();
  }
  
  public void verifyConvergence(final int maxIter, final double convergence, final int reps) {
    final long succeesses = IntStream.range(0, reps) //
        .parallel()
        .filter(i -> {
          boolean hasConverged = false;
          try {
            final MacroTrainer copy = Util.kryo().copy(this.macroTrainer);
            final Double error = copy.setMaxIterations(maxIter).setStopError(convergence).train();
            hasConverged = error <= convergence;
            if (!hasConverged) {
              Trainer.log.debug("Not Converged");
            }
          } catch (final Throwable e) {
            Trainer.log.debug("Not Converged", e);
          }
          return hasConverged;
        }).count();
    if (reps > succeesses) throw new RuntimeException(String.format("%s out of %s converged", succeesses, reps));
  }
  
}
