# NormalizationMetaLayer
## NormalizationMetaLayerTest
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.24 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.200.png)



### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ],
    [ -1.648, -1.124, -1.212 ]
    Inputs Statistics: {meanExponent=0.11707537947480223, negative=3, min=-1.212, max=-1.212, mean=-1.328, count=3.0, positive=0, stdDev=0.22910841683942176, zeros=0},
    {meanExponent=0.11707537947480223, negative=3, min=-1.212, max=-1.212, mean=-1.328, count=3.0, positive=0, stdDev=0.22910841683942176, zeros=0},
    {meanExponent=0.11707537947480223, negative=3, min=-1.212, max=-1.212, mean=-1.328, count=3.0, positive=0, stdDev=0.22910841683942176, zeros=0},
    {meanExponent=0.11707537947480223, negative=3, min=-1.212, max=-1.212, mean=-1.328, count=3.0, positive=0, stdDev=0.22910841683942176, zeros=0},
    {meanExponent=0.11707537947480223, negative=3, min=-1.212, max=-1.212, mean=-1.328, count=3.0, positive=0, stdDev=0.22910841683942176, ze
```
...[skipping 1493 bytes](etc/295.txt)...
```
    {meanExponent=-0.5254973270962222, negative=6, min=0.541979292891704, max=0.541979292891704, mean=0.007149246698753053, count=9.0, positive=3, stdDev=0.34973264396055903, zeros=0}
    Measured: [ [ 0.7050624604842959, -0.025228446421943218, -0.027203627280814047 ], [ -0.025228053179837673, 0.7248449854146966, -0.018553641052410086 ], [ -0.027203297707778518, -0.01855370547643176, 0.7220453085188261 ] ]
    Measured Statistics: {meanExponent=-1.135897821618769, negative=6, min=0.7220453085188261, max=0.7220453085188261, mean=0.22333133147762257, count=9.0, positive=3, stdDev=0.34935055055308095, zeros=0}
    Feedback Error: [ [ 0.33291954479254743, 0.2270625276101767, 0.24483966500293075 ], [ 0.22706292085228208, 0.15486726082651447, 0.16699044907315375 ], [ 0.24483999457596628, 0.16699038464913216, 0.1800660156271221 ] ]
    Error Statistics: {meanExponent=-0.67742560136677, negative=0, min=0.1800660156271221, max=0.1800660156271221, mean=0.21618208477886952, count=9.0, positive=9, stdDev=0.053135531500832646, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=2.1618e-01 +- 5.3136e-02 [1.5487e-01 - 3.3292e-01] (9#), relativeTol=6.0892e-01 +- 2.9994e-01 [1.1960e-01 - 8.1819e-01] (9#)}
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.lambda$test$9(BatchDerivativeTester.java:107)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.test(BatchDerivativeTester.java:138)
    	at com.simiacryptu
```
...[skipping 2930 bytes](etc/296.txt)...
```
    unner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



