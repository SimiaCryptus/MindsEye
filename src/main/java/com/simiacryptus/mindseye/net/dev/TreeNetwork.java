package com.simiacryptus.mindseye.net.dev;

import java.util.ArrayList;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.util.Util;

public final class TreeNetwork extends DAGNetwork {
  
  private final java.util.List<NNLayer<?>> gates = new java.util.ArrayList<>();
  private final int[] inputSize;
  private final java.util.List<WrapperLayer> leafs = new java.util.ArrayList<>();
  private final int[] outSize;

  public TreeNetwork(final int[] inputSize, final int[] outSize) {
    this.inputSize = inputSize;
    this.outSize = outSize;
    add(nodeFactory());
  }

  public DAGNetwork constFactory(final int i) {
    DAGNetwork subnet = new DAGNetwork();
    subnet = subnet.add(new DenseSynapseLayer(NDArray.dim(this.inputSize), this.outSize).setWeights(() -> 0).freeze());
    subnet = subnet.add(new BiasLayer(this.outSize).setWeights(j -> i == j ? 1 : 0).freeze());
    return subnet;
  }

  @Override
  public TreeNetwork evolve() {
    this.gates.stream().forEach(l -> l.freeze());
    final ArrayList<WrapperLayer> lcpy = new java.util.ArrayList<>(this.leafs);
    this.leafs.clear();
    for (final WrapperLayer l : lcpy) {
      l.setInner(nodeFactory());
    }
    return this;
  }

  public DAGNetwork gateFactory() {
    DAGNetwork gate = new DAGNetwork();
    gate = gate.add(new DenseSynapseLayer(NDArray.dim(this.inputSize), new int[] { 2 }).setWeights(()->Util.R.get().nextGaussian()).setVerbose(true));
    gate = gate.add(new BiasLayer(new int[] { 2 }).setVerbose(true));
    gate = gate.add(new SoftmaxActivationLayer().setVerbose(true));
    this.gates.add(gate);
    return gate;
  }

  public WrapperLayer leafFactory(final int i) {
    final DAGNetwork subnet = constFactory(i);
    final WrapperLayer wrapper = new WrapperLayer(subnet);
    this.leafs.add(wrapper);
    return wrapper;
  }

  public TreeNodeFunctionalLayer nodeFactory() {
    return new TreeNodeFunctionalLayer(gateFactory(), 2, this::leafFactory);
  }
}
