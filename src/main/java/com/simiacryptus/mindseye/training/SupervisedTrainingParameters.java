package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;

public class SupervisedTrainingParameters {
  private PipelineNetwork net;
  private final NDArray[][] trainingData;

  protected SupervisedTrainingParameters() {
    super();
    this.net = null;
    this.trainingData = null;
  }
  
  public SupervisedTrainingParameters(final PipelineNetwork net, final NDArray[][] trainingData) {
    this.net = net;
    this.trainingData = trainingData;
  }

  public NDArray interceptIdeal(final NNResult eval, final NDArray preset) {
    return preset;
  }

  public PipelineNetwork getNet() {
    return this.net;
  }

  public final NDArray[][] getTrainingData() {
    return this.trainingData;
  }
  
  public void setNet(final PipelineNetwork net) {
    this.net = net;
  }
  
}