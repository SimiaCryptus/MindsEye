package com.simiacryptus.mindseye.training.dev;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.TrainingComponent;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.training.TrainingComponent.TrainingStep;

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
      this.inner.reset();
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
  public TrainingStep step(final TrainingContext trainingContext) {
    double prevError = getError();
    int lifecycle = 0;
    double endError;
    do {
      TrainingStep step = inner.step(trainingContext);
      if(!Double.isFinite(prevError)) prevError = step.getStartError();
      endError = step.finalError();
    } while (lifecycle++ < getEvolutionPhases() && evolve(trainingContext));
    // train2(trainingContext);
    return new TrainingStep(prevError, endError, true);
  }


  public NDArray[][] getData() {
    return inner.getData();
  }

  @Override
  public void reset() {
    inner.reset();
  }
}
