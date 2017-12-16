# PoolingLayer
## Float
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (400#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.544, -1.812 ], [ 0.068, 1.444 ], [ -0.852, -1.38 ], [ 1.864, -1.812 ] ],
    	[ [ 1.704, 0.68 ], [ -0.708, -1.624 ], [ 0.484, 0.028 ], [ 1.82, 0.628 ] ],
    	[ [ 1.48, 0.704 ], [ -0.164, -0.736 ], [ 0.996, 0.268 ], [ 1.564, 1.504 ] ],
    	[ [ -0.484, -0.224 ], [ 1.48, 0.436 ], [ 1.512, 0.228 ], [ -1.5, -1.392 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.14065994801754808, negative=13, min=-1.392, max=-1.392, mean=0.1768749999999999, count=32.0, positive=19, stdDev=1.155419722168096, zeros=0}
    Output: [
    	[ [ 1.704, 1.444 ], [ 1.864, 0.628 ] ],
    	[ [ 1.48, 0.704 ], [ 1.564, 1.504 ] ]
    ]
    Outputs Statistics: {meanExponent=0.10609516186699781, negative=0, min=1.504, max=1.504, mean=1.3615000000000002, count=8.0, positive=8, stdDev=0.42164884679078585, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.544, -1.812 ], [ 0.068, 1.444 ], [ -0.852, -1.38 ], [ 1.864, -1.812 ] ],
    	[ [ 1.704, 0.68 ], [ -0.708, -1.624 ], [ 0.484, 0.028 ], [ 1.82, 0.628 ] ],
    	[ [ 1.48, 0.704 ], [ -0.164, -0.736 ], [ 0.996, 0.268 
```
...[skipping 996 bytes](etc/132.txt)...
```
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=-3.7115892712675236E-14, negative=0, min=0.0, max=0.0, mean=0.035156249999996995, count=256.0, positive=9, stdDev=0.18417461303320742, zeros=247}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=-11.983496461624346, negative=6, min=0.0, max=0.0, mean=0.0039062499999969955, count=256.0, positive=3, stdDev=0.0623778102448031, zeros=247}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.9063e-03 +- 6.2378e-02 [0.0000e+00 - 1.0000e+00] (256#), relativeTol=1.1111e-01 +- 3.1427e-01 [4.4409e-16 - 1.0000e+00] (9#)}
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.lambda$test$7(SingleDerivativeTester.java:107)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.test(SingleDerivativeTester.java:138)
    	at com.simia
```
...[skipping 2850 bytes](etc/133.txt)...
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



