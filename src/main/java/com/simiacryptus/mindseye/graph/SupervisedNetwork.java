package com.simiacryptus.mindseye.graph;

import com.simiacryptus.mindseye.graph.dag.DAGNetwork;

public abstract class SupervisedNetwork extends DAGNetwork {
    public SupervisedNetwork(int inputs) {
        super(inputs);
    }
}
