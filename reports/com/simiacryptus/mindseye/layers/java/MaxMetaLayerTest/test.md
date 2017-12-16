# MaxMetaLayer
## MaxMetaLayerTest
### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ],
    [ 1.264, 1.072, -1.332 ]
    Inputs Statistics: {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positive=2, stdDev=1.1811151039965957, zeros=0},
    {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positive=2, stdDev=1.1811151039965957, zeros=0},
    {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positive=2, stdDev=1.1811151039965957, zeros=0},
    {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positive=2, stdDev=1.1811151039965957, zeros=0},
    {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positi
```
...[skipping 2263 bytes](etc/285.txt)...
```
    Statistics: {meanExponent=0.08548202804579992, negative=1, min=-1.332, max=-1.332, mean=0.3346666666666667, count=3.0, positive=2, stdDev=1.1811151039965957, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Measured: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Error Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.3333e-01 +- 4.7140e-01 [0.0000e+00 - 1.0000e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
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
...[skipping 2930 bytes](etc/286.txt)...
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



