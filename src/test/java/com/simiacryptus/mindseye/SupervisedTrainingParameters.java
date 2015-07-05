package com.simiacryptus.mindseye;

public class SupervisedTrainingParameters {
  private PipelineNetwork net;
  private NDArray[][] trainingData;
  
  protected SupervisedTrainingParameters() {
    super();
  }

  public SupervisedTrainingParameters(PipelineNetwork net, NDArray[][] trainingData) {
    this.net = net;
    this.trainingData = trainingData;
  }
  
  public PipelineNetwork getNet() {
    return net;
  }
  
  public void setNet(PipelineNetwork net) {
    this.net = net;
  }
  
  public NDArray[][] getTrainingData() {
    return trainingData;
  }
}