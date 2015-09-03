package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.math.NDArray;

public class SupervisedTrainingParameters {
  private PipelineNetwork net;
  private NDArray[][] trainingData;

  protected SupervisedTrainingParameters() {
    super();
    this.net = null;
    this.trainingData = null;
  }
  
  public SupervisedTrainingParameters(final PipelineNetwork net, final NDArray[][] trainingData) {
    this.net = net;
    this.trainingData = trainingData;
  }

  public PipelineNetwork getNet() {
    return this.net;
  }

  public final NDArray[][] getTrainingData() {
    return this.trainingData;
  }
  
  public SupervisedTrainingParameters setNet(final PipelineNetwork net) {
    this.net = net;
    return this;
  }

  public SupervisedTrainingParameters setTrainingData(NDArray[][] trainingData) {
    this.trainingData = trainingData;
    return this;
  }
  
}