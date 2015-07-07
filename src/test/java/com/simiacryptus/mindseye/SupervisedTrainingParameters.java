package com.simiacryptus.mindseye;

public class SupervisedTrainingParameters {
  private double weight = 1;
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

  public double getWeight() {
    return weight;
  }

  public SupervisedTrainingParameters setWeight(double weight) {
    this.weight = weight;
    return this;
  }
}