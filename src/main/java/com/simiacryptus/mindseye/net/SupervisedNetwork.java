package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dag.DAGNode;

public abstract class SupervisedNetwork extends DAGNetwork {
    public SupervisedNetwork(int inputs) {
        super(inputs);
    }
}
