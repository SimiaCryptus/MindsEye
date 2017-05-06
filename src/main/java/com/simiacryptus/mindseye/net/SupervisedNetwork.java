package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dag.DAGNode;

public class SupervisedNetwork extends DAGNetwork {


    private final DAGNode studentNode;
    private final DAGNode lossNode;

    public SupervisedNetwork(final NNLayer<?> student, final NNLayer<?> loss) {
        super(2);
        studentNode = add(student, getInput(0));
        lossNode = add(loss, studentNode, getInput(1));
    }

    @Override
    public DAGNode getHead() {
        return lossNode;
    }
}
