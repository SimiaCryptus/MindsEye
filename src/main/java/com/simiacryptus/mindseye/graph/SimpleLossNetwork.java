package com.simiacryptus.mindseye.graph;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.graph.dag.DAGNode;

public class SimpleLossNetwork extends SupervisedNetwork {


    public final DAGNode studentNode;
    public final DAGNode lossNode;

    public SimpleLossNetwork(final NNLayer student, final NNLayer loss) {
        super(2);
        studentNode = add(student, getInput(0));
        lossNode = add(loss, studentNode, getInput(1));
    }

    @Override
    public DAGNode getHead() {
        return lossNode;
    }
}
