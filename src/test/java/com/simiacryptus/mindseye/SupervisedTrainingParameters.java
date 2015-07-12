package com.simiacryptus.mindseye;

public class SupervisedTrainingParameters {
  private double weight = 1;
  private PipelineNetwork net;
  private final NDArray[][] trainingData;
  
  protected SupervisedTrainingParameters() {
    super();
    this.net = null;
    this.trainingData = null;
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
  
  public final NDArray[][] getTrainingData() {
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