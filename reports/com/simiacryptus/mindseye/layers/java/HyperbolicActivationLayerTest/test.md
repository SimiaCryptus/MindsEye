# HyperbolicActivationLayer
## HyperbolicActivationLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.6 ], [ 0.928 ], [ -1.704 ] ],
    	[ [ -0.012 ], [ -1.844 ], [ -0.408 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.3445381428519496, negative=4, min=-0.408, max=-0.408, mean=-0.4066666666666667, count=6.0, positive=2, stdDev=1.0566832806264872, zeros=0}
    Output: [
    	[ [ 0.16619037896905997 ], [ 0.36425217610235094 ], [ 0.9757570700873122 ] ],
    	[ [ 7.199740818664147E-5 ], [ 1.097697785668851 ], [ 0.08002962922319856 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.0712666340788473, negative=0, min=0.08002962922319856, max=0.08002962922319856, mean=0.4473331729098266, count=6.0, positive=6, stdDev=0.43264768568444634, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.6 ], [ 0.928 ], [ -1.704 ] ],
    	[ [ -0.012 ], [ -1.844 ], [ -0.408 ] ]
    ]
    Value Statistics: {meanExponent=-0.3445381428519496, negative=4, min=-0.408, max=-0.408, mean=-0.4066666666666667, count=6.0, positive=2, stdDev=1.0566832806264872, zeros=0}
    Implemented Feedback: [ [ 0.5144957554275266, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.01199913
```
...[skipping 1918 bytes](etc/237.txt)...
```
    2.0, positive=0, stdDev=0.40065398539201075, zeros=6}
    Measured Gradient: [ [ -0.85739583725708, 0.0, -0.7329120638527309, 0.0, 0.0, 0.0 ], [ 0.0, -0.9998280177740633, 0.0, -0.47664701104732643, -0.5060656698441246, -0.9258013376195473 ] ]
    Measured Statistics: {meanExponent=-0.14215331692749142, negative=6, min=-0.9258013376195473, max=-0.9258013376195473, mean=-0.374887494782906, count=12.0, positive=0, stdDev=0.40060986085287165, zeros=6}
    Gradient Error: [ [ 9.708845546430034E-5, 0.0, 9.024870870899537E-5, 0.0, 0.0, 0.0 ], [ 0.0, 9.999000100369138E-5, 0.0, 6.60821502954656E-5, 6.942901993167716E-5, 9.918671999409323E-5 ] ]
    Error Statistics: {meanExponent=-4.066559351417276, negative=0, min=9.918671999409323E-5, max=9.918671999409323E-5, mean=4.350208794985192E-5, count=12.0, positive=6, stdDev=4.461414467069767E-5, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4059e-05 +- 2.9803e-05 [0.0000e+00 - 9.9990e-05] (48#)
    relativeTol: 2.1263e-04 +- 5.6572e-04 [3.0811e-06 - 2.0874e-03] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4059e-05 +- 2.9803e-05 [0.0000e+00 - 9.9990e-05] (48#), relativeTol=2.1263e-04 +- 5.6572e-04 [3.0811e-06 - 2.0874e-03] (12#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.HyperbolicActivationLayer",
      "id": "cb4d1333-2e05-4365-87d4-d486bcf2d61a",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/cb4d1333-2e05-4365-87d4-d486bcf2d61a",
      "weights": [
        1.0,
        1.0
      ],
      "negativeMode": 1
    }
```



### Reference Input/Output Pairs
Code from [ReferenceIO.java:56](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L56) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, input);
    DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s\n--------------------\nDerivative: \n%s",
      Arrays.stream(input).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(), error,
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.0 ]]
    --------------------
    Output: 
    [ 0.0 ]
    Error: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    --------------------
    Derivative: 
    [ 0.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.796 ], [ 0.952 ], [ -0.644 ], [ 0.676 ], [ -1.84 ], [ -1.964 ], [ 1.132 ], [ 1.412 ], ... ],
    	[ [ -0.648 ], [ -1.356 ], [ -0.964 ], [ 0.936 ], [ -0.78 ], [ -1.932 ], [ 0.244 ], [ -1.18 ], ... ],
    	[ [ -0.544 ], [ 0.552 ], [ 0.0 ], [ -0.9 ], [ -0.124 ], [ -0.196 ], [ 1.508 ], [ -1.284 ], ... ],
    	[ [ 0.828 ], [ 1.332 ], [ -1.16 ], [ -1.452 ], [ -0.556 ], [ -0.488 ], [ 1.14 ], [ 1.26 ], ... ],
    	[ [ 0.612 ], [ -1.008 ], [ 0.984 ], [ -0.6 ], [ 1.28 ], [ -1.516 ], [ -0.168 ], [ 0.896 ], ... ],
    	[ [ -1.308 ], [ 0.18 ], [ -0.7 ], [ -1.032 ], [ 1.448 ], [ -1.356 ], [ 1.352 ], [ 1.492 ], ... ],
    	[ [ 0.42 ], [ -0.076 ], [ -0.116 ], [ 1.592 ], [ -0.412 ], [ 0.324 ], [ -0.684 ], [ -1.416 ], ... ],
    	[ [ -1.084 ], [ 1.556 ], [ -1.196 ], [ 0.016 ], [ 1.16 ], [ 1.684 ], [ -0.696 ], [ 0.324 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 3.27 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.2969373597888623}, derivative=-5.487949018836758E-5}
    New Minimum: 0.2969373597888623 > 0.29693735978885705
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=0.29693735978885705}, derivative=-5.4879490188366854E-5}, delta = -5.2735593669694936E-15
    New Minimum: 0.29693735978885705 > 0.2969373597888241
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=0.2969373597888241}, derivative=-5.487949018836252E-5}, delta = -3.824718319833664E-14
    New Minimum: 0.2969373597888241 > 0.2969373597885927
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=0.2969373597885927}, derivative=-5.4879490188332146E-5}, delta = -2.6961766153021927E-13
    New Minimum: 0.2969373597885927 > 0.296937359786979
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=0.296937359786979}, derivative=-5.487949018811956E-5}, delta = -1.8833268278228843E-12
    New Minimum: 0.296937359786979 > 0.2969373597756866
    F(2.40100
```
...[skipping 323067 bytes](etc/238.txt)...
```
    389487919158813E-20}, delta = -1.4474310693155662E-10
    Left bracket at 12532.275584677991
    Converged to left
    Iteration 249 complete. Error: 4.0774493264565545E-4 Total: 239660121610544.9000; Orientation: 0.0004; Line Search: 0.0132
    Low gradient: 1.5164579378306255E-7
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.0774493264565545E-4}, derivative=-2.299644677209513E-14}
    New Minimum: 4.0774493264565545E-4 > 4.077447902145846E-4
    F(12532.275584677991) = LineSearchPoint{point=PointSample{avg=4.077447902145846E-4}, derivative=2.663773908205842E-16}, delta = -1.4243107084380147E-10
    4.077447902145846E-4 <= 4.0774493264565545E-4
    New Minimum: 4.077447902145846E-4 > 4.077447901954713E-4
    F(12388.771303000412) = LineSearchPoint{point=PointSample{avg=4.077447901954713E-4}, derivative=-7.182591754749818E-21}, delta = -1.4245018413548E-10
    Left bracket at 12388.771303000412
    Converged to left
    Iteration 250 complete. Error: 4.077447901954713E-4 Total: 239660127700817.8800; Orientation: 0.0003; Line Search: 0.0049
    
```

Returns: 

```
    4.077447901954713E-4
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -1.7959999999999818 ], [ 0.952 ], [ -0.6439999999999999 ], [ -0.6759999999999998 ], [ -1.8399999999979173 ], [ 1.964004443506618 ], [ 1.1319999999999997 ], [ 1.4120000000000001 ], ... ],
    	[ [ 0.6479999999999999 ], [ 1.3560000000000008 ], [ -0.9639999999999997 ], [ -0.936 ], [ 0.7799999999999998 ], [ 1.9319999278208346 ], [ -0.24400000000000027 ], [ 1.1800000000000002 ], ... ],
    	[ [ 0.5439999999999999 ], [ 0.5520000000000002 ], [ 0.039084754552563376 ], [ 0.9 ], [ -0.12399974932485446 ], [ 0.19600000000337844 ], [ -1.5080000000000002 ], [ -1.2839999999999994 ], ... ],
    	[ [ 0.8279999999999998 ], [ -1.3320000000000003 ], [ -1.1600000000000006 ], [ -1.452 ], [ -0.556 ], [ -0.4879999999999997 ], [ -1.1400000000000006 ], [ -1.2600000000000002 ], ... ],
    	[ [ -0.6119999999999999 ], [ -1.0079999999999998 ], [ -0.9839999999999999 ], [ -0.5999999999999999 ], [ -1.28 ], [ 1.5159999999999998 ], [ 0.16799997652204374 ], [ -0.8960000000000001 ], ... ],
    	[ [ 1.308 ], [ -0.180000000133645 ], [ -0.7000000000000001 ], [ 1.0320000000000003 ], [ 1.448 ], [ 1.3559999999999994 ], [ -1.3520000000000003 ], [ 1.4919999999999998 ], ... ],
    	[ [ 0.4200000000000004 ], [ 0.07705757253422686 ], [ -0.11601288871162939 ], [ -1.592 ], [ -0.41200000000000025 ], [ -0.3240000000000004 ], [ 0.684 ], [ 1.4160000000000001 ], ... ],
    	[ [ 1.084 ], [ 1.556 ], [ 1.1960000000000002 ], [ -0.04124736182831454 ], [ 1.1600000000000006 ], [ 1.6839999999999997 ], [ 0.696 ], [ 0.3240000000000004 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.03 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=0.2969373597888623;dx=-5.487949018836758E-5
    New Minimum: 0.2969373597888623 > 0.2968191422927981
    WOLFE (weak): th(2.154434690031884)=0.2968191422927981; dx=-5.486391173806947E-5 delta=1.1821749606422127E-4
    New Minimum: 0.2968191422927981 > 0.2967009583595359
    WOLFE (weak): th(4.308869380063768)=0.2967009583595359; dx=-5.4848333242738235E-5 delta=2.3640142932640495E-4
    New Minimum: 0.2967009583595359 > 0.2962285582564155
    WOLFE (weak): th(12.926608140191302)=0.2962285582564155; dx=-5.4786018824661475E-5 delta=7.088015324468455E-4
    New Minimum: 0.2962285582564155 > 0.29410940339608616
    WOLFE (weak): th(51.70643256076521)=0.29410940339608616; dx=-5.450559632979501E-5 delta=0.0028279563927761653
    New Minimum: 0.29410940339608616 > 0.2829909135399678
    WOLFE (weak): th(258.53216280382605)=0.2829909135399678; dx=-5.300994837298864E-5 delta=0.013946446248894517
    New Minimum: 0.2829909135399678 > 0.22049653372426764
    END: th(1551.1
```
...[skipping 202 bytes](etc/239.txt)...
```
    04
    LBFGS Accumulation History: 1 points
    th(0)=0.22049653372426764;dx=-3.549194954520103E-5
    New Minimum: 0.22049653372426764 > 0.12386003579747712
    END: th(3341.943960201201)=0.12386003579747712; dx=-2.2325886653351557E-5 delta=0.09663649792679052
    Iteration 2 complete. Error: 0.12386003579747712 Total: 239660156069846.8400; Orientation: 0.0005; Line Search: 0.0026
    LBFGS Accumulation History: 1 points
    th(0)=0.12386003579747712;dx=-1.5808759289167678E-5
    New Minimum: 0.12386003579747712 > 0.04379740625045247
    END: th(7200.000000000001)=0.04379740625045247; dx=-5.791627360840347E-6 delta=0.08006262954702464
    Iteration 3 complete. Error: 0.04379740625045247 Total: 239660160194908.8400; Orientation: 0.0005; Line Search: 0.0029
    LBFGS Accumulation History: 1 points
    th(0)=0.04379740625045247;dx=-4.311312384221667E-6
    MAX ALPHA: th(0)=0.04379740625045247;th'(0)=-4.311312384221667E-6;
    Iteration 4 failed, aborting. Error: 0.04379740625045247 Total: 239660163917580.8400; Orientation: 0.0005; Line Search: 0.0024
    
```

Returns: 

```
    0.04379740625045247
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.05 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.09120164751894672 ], [ 0.608148966066444 ], [ -0.5479216848133107 ], [ -0.5056663820531094 ], [ -0.8130898951795167 ], [ 1.7804509501047228 ], [ 1.1360315106367844 ], [ 1.412506168337367 ], ... ],
    	[ [ 0.7698511692226636 ], [ 0.056130026178692574 ], [ -1.0125940769858348 ], [ -0.5671631581690775 ], [ 0.8582187632091064 ], [ 1.9499450055621952 ], [ -0.5483931685198957 ], [ 0.9009515225579431 ], ... ],
    	[ [ 0.7312705153604335 ], [ 0.6125172190373095 ], [ 0.16253500150558045 ], [ 0.1055783855093799 ], [ -0.582731291688719 ], [ 0.47799636061371226 ], [ -1.4303427582354107 ], [ -0.6630452585852236 ], ... ],
    	[ [ 0.874554905530612 ], [ -1.33446991463809 ], [ -0.08898218232487051 ], [ -0.3809133604618955 ], [ -0.6520053064108998 ], [ -0.2881317486538865 ], [ -0.6612447574616311 ], [ -1.2672402993207519 ], ... ],
    	[ [ -0.29666029776584396 ], [ -1.0387141811747296 ], [ -1.020726297378315 ], [ -0.7518704965889724 ], [ -0.19103795037564955 ], [ 1.5033084154598382 ], [ 0.5928522479206769 ], [ -0.950291621737141 ], ... ],
    	[ [ 1.3122451580421466 ], [ -0.40965644863568257 ], [ -0.6029373403382201 ], [ 0.5017361578270946 ], [ 1.3757961317318108 ], [ 1.3480694775715187 ], [ -1.2164510702412876 ], [ 1.489141515462558 ], ... ],
    	[ [ 0.6026247481222937 ], [ 0.48734852679137164 ], [ -0.5424273552595398 ], [ -1.5490735216149325 ], [ -0.5094894581471668 ], [ -0.5346683814539306 ], [ 0.5131886895068618 ], [ 1.2589310173109773 ], ... ],
    	[ [ 1.0800195580977305 ], [ 1.5557446000176491 ], [ 1.202312280957149 ], [ -0.5660992971973705 ], [ 1.1664108809557803 ], [ 0.9321135903720079 ], [ 0.640520080912477 ], [ 0.5915236514494661 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.150.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.151.png)



### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.0, 1.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.9310888120436096, 0.012094857214480559, 0.5712873702795425, 1.196801310997424, 0.03126330294450019, 0.14808362064790392, 0.3241691734819989, 0.6784230694315425, 0.9140073145105794, 0.8227934605983203, 0.7172116934146471, 0.02222502415074934, 1.0451542729095036, 0.9447940765027025, 1.0626235720557449, 0.425572165833775, 0.26088223082094397, 0.8969828676084559, 0.7042300314218148, 0.03846039885977359, 0.5406440211807528, 0.060143386528445086, 0.5589791531640185, 0.05367167561816899, 0.03324730824715916, 0.5743747965462354, 1.196801310997424, 0.6054706475049614, 0.23223374405994912, 0.20481699855206226, 0.12712022428842973, 0.9345118247247806, 0.02059590436176073, 1.1861235097770666, 0.972308292331602, 0.5961002474782091, 0.0011513372113129972, 0.024780952203933282, 0.9345118247247806, 0.5743747965462354, 1.034695063148284, 0.7237215552402889, 0.8027756377319946, 0.10579564115617668, 0.9585341457324659, 0.06556276211211509, 0.40294547292473193, 0.40575389026671393, 0.6243152403397563, 0.993031861260627, 0.796
```
...[skipping 202190 bytes](etc/240.txt)...
```
    475418846041431, 0.659209450310599, 0.24880102498356393, 0.8597376159017702, 0.024780952203933282, 0.22990080900859633, 0.6401219466856727, 0.789484842070477, 0.6086018774078314, 0.6337882359718472, 1.193240524885495, 0.15203472169895993, 0.38068968273106174, 0.48956906519973087, 1.0381795799192965, 0.7009926513656666, 0.20481699855206226, 0.17450244784759805, 0.034261088893902114, 0.07555381083421397, 0.8969828676084559, 0.37518580562773396, 1.2003636063160106, 0.08154704012354452, 0.2781298838537498, 0.3697065379124098, 0.06282642044691378, 0.026559301745398445, 0.6369532675064369, 0.08002962922319856, 0.1216345215800021, 6.477901839387901E-4, 0.9413644686147935, 5.118689950658339E-4, 0.1327064933158988, 0.209297316626478, 1.0801384569302113, 1.161262593948269, 0.216085523308291, 0.31114606356423913, 0.8732474476160377, 1.0801384569302113, 0.4398944405754195, 1.225341322134652, 0.507442867905779, 0.12712022428842973, 0.22757647419621074, 0.8698663053812163, 0.9585341457324659, 0.7172116934146471]
    [1.0, 1.0]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:203](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.9310888120436096, 0.012094857214480559, 0.5712873702795425, 1.196801310997424, 0.03126330294450019, 0.14808362064790392, 0.3241691734819989, 0.6784230694315425, 0.9140073145105794, 0.8227934605983203, 0.7172116934146471, 0.02222502415074934, 1.0451542729095036, 0.9447940765027025, 1.0626235720557449, 0.425572165833775, 0.26088223082094397, 0.8969828676084559, 0.7042300314218148, 0.03846039885977359, 0.5406440211807528, 0.060143386528445086, 0.5589791531640185, 0.05367167561816899, 0.03324730824715916, 0.5743747965462354, 1.196801310997424, 0.6054706475049614, 0.23223374405994912, 0.20481699855206226, 0.12712022428842973, 0.9345118247247806, 0.02059590436176073, 1.1861235097770666, 0.972308292331602, 0.5961002474782091, 0.0011513372113129972, 0.024780952203933282, 0.9345118247247806, 0.5743747965462354, 1.034695063148284, 0.7237215552402889, 0.8027756377319946, 0.10579564115617668, 0.9585341457324659, 0.06556276211211509, 0.40294547292473193, 0.40575389026671393, 0.6243152403397563, 0.993031861260627, 0.796
```
...[skipping 202190 bytes](etc/241.txt)...
```
    475418846041431, 0.659209450310599, 0.24880102498356393, 0.8597376159017702, 0.024780952203933282, 0.22990080900859633, 0.6401219466856727, 0.789484842070477, 0.6086018774078314, 0.6337882359718472, 1.193240524885495, 0.15203472169895993, 0.38068968273106174, 0.48956906519973087, 1.0381795799192965, 0.7009926513656666, 0.20481699855206226, 0.17450244784759805, 0.034261088893902114, 0.07555381083421397, 0.8969828676084559, 0.37518580562773396, 1.2003636063160106, 0.08154704012354452, 0.2781298838537498, 0.3697065379124098, 0.06282642044691378, 0.026559301745398445, 0.6369532675064369, 0.08002962922319856, 0.1216345215800021, 6.477901839387901E-4, 0.9413644686147935, 5.118689950658339E-4, 0.1327064933158988, 0.209297316626478, 1.0801384569302113, 1.161262593948269, 0.216085523308291, 0.31114606356423913, 0.8732474476160377, 1.0801384569302113, 0.4398944405754195, 1.225341322134652, 0.507442867905779, 0.12712022428842973, 0.22757647419621074, 0.8698663053812163, 0.9585341457324659, 0.7172116934146471]
    [1.0, 1.0]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.44 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.020047s +- 0.001125s [0.018383s - 0.021551s]
    	Learning performance: 0.036477s +- 0.009621s [0.026990s - 0.051555s]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.152.png)



Code from [ActivationLayerTestBase.java:114](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L114) executed in 0.00 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.153.png)



