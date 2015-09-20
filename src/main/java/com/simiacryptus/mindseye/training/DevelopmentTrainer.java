package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;

public class DevelopmentTrainer implements TrainingComponent {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DevelopmentTrainer.class);

  private int evolutionPhases = 2;
  private final TrainingComponent inner;
  private boolean verbose = false;
  
  public DevelopmentTrainer(TrainingComponent inner) {
    super();
    this.inner = inner;
  }

  protected DevelopmentTrainer() {
    super();
    inner = new GradientDescentTrainer();
  }

  private boolean evolve(final TrainingContext trainingContext) {
    final boolean isValid = null != this.inner.getNet().evolve();
    if (isValid) {
      this.inner.refresh();
      this.inner.step(trainingContext);
    }
    return isValid;
  }

  @Override
  public double getError() {
    return this.inner.getError();
  }

  public int getEvolutionPhases() {
    return this.evolutionPhases;
  }

  @Override
  public DAGNetwork getNet() {
    return this.inner.getNet();
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public TrainingComponent setEvolutionPhases(final int evolutionPhases) {
    this.evolutionPhases = evolutionPhases;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public double step(final TrainingContext trainingContext) {
    int lifecycle = 0;
    do {
      inner.step(trainingContext);
    } while (lifecycle++ < getEvolutionPhases() && evolve(trainingContext));
    // train2(trainingContext);
    return getError();
  }


  public NDArray[][] getData() {
    return inner.getData();
  }

  @Override
  public void refresh() {
    inner.refresh();
  }
}
