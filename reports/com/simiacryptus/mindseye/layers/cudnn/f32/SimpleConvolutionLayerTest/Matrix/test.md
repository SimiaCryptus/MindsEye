# SimpleConvolutionLayer
## Matrix
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000437",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000000437",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.116,
          0.272,
          -1.1,
          0.184,
          0.824,
          -0.164,
          -0.64,
          -1.244,
          -0.936
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -0.076 ], [ 1.088 ], [ -1.232 ] ],
    	[ [ -0.976 ], [ 1.108 ], [ -1.66 ] ],
    	[ [ 0.388 ], [ 0.736 ], [ -1.212 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.07480049133300781 ], [ 1.677024006843567 ], [ -3.383199453353882 ] ],
    	[ [ -1.7011685371398926 ], [ 2.9512157440185547 ], [ -4.256559371948242 ] ],
    	[ [ -0.5388323068618774 ], [ 2.351951837539673 ], [ -2.67911958694458 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000438",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000438",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -0.116,
          0.272,
          -1.1,
          0.184,
          0.824,
          -0.164,
          -0.64,
          -1.244,
          -0.936
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ -0.076 ], [ 1.088 ], [ -1.232 ] ],
    	[ [ -0.976 ], [ 1.108 ], [ -1.66 ] ],
    	[ [ 0.388 ], [ 0.736 ], [ -1.212 ] ]
    ]
    Error: [
    	[ [ -4.913330078210043E-7 ], [ 6.8435668243438386E-9 ], [ 5.466461181491411E-7 ] ],
    	[ [ -5.371398925646531E-7 ], [ -2.5598144581806537E-7 ], [ 6.280517572676558E-7 ] ],
    	[ [ -3.068618772417153E-7 ], [ -1.6246032741307204E-7 ], [ 4.130554200898473E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 3.7204e-07 +- 1.9270e-07 [6.8436e-09 - 6.2805e-07] (9#)
    relativeTol: 4.4872e-07 +- 1.0056e-06 [2.0404e-09 - 3.2843e-06] (9#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5686e-07 +- 2.7092e-07 [0.0000e+00 - 1.4305e-06] (180#), relativeTol=1.4547e-07 +- 8.6961e-07 [0.0000e+00 - 1.1346e-05] (180#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.076 ], [ 1.088 ], [ -1.232 ] ],
    	[ [ -0.976 ], [ 1.108 ], [ -1.66 ] ],
    	[ [ 0.388 ], [ 0.736 ], [ -1.212 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13318188651861218, negative=5, min=-1.212, max=-1.212, mean=-0.20400000000000001, count=9.0, positive=4, stdDev=1.0236006165601026, zeros=0}
    Output: [
    	[ [ -0.07480049133300781 ], [ 1.677024006843567 ], [ -3.383199453353882 ] ],
    	[ [ -1.7011685371398926 ], [ 2.9512157440185547 ], [ -4.256559371948242 ] ],
    	[ [ -0.5388323068618774 ], [ 2.351951837539673 ], [ -2.67911958694458 ] ]
    ]
    Outputs Statistics: {meanExponent=0.16538360879778866, negative=6, min=-2.67911958694458, max=-2.67911958694458, mean=-0.6281653510199653, count=9.0, positive=3, stdDev=2.4366270792224585, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.076 ], [ 1.088 ], [ -1.232 ] ],
    	[ [ -0.976 ], [ 1.108 ], [ -1.66 ] ],
    	[ [ 0.388 ], [ 0.736 ], [ -1.212 ] ]
    ]
    Value Statistics: {meanExponent=-0.13318188651861218, negative=5, min=-1.212, max=-1.212, mean=-0.20400000000000001, count=9.0, positive=4, stdDev=1.0236006165601026, zeros=0}
    Implemented Feedback: [ [ 0.8239999413490295, -0.16400009393692017, 0.0, -1.24399995803833, -0.9359999895095825, 0.0, 0.0, 0.0, 0.0 ], [ 0.1839999556541443, 0.8239999413490295, -0.1640000343322754, -0.6399999856948853, -1.24399995803833, -0.9359999895095825, 0.0, 0.0, 0.0 ], [ 0.0, 0.18400001525878906, 0.8239999413490295, 0.0, -0.6399999856948853, -1.24399995803833, 0.0, 0.0, 0.0 ], [ 0.2720000147819519, -1.100000023841858, 0.0, 0.8239999413490295, -0.16400009393692017, 0.0, -1.24399995803833, -0.9359999895095825, 0.0 ], [ -0.11600005626678467, 0.2720000147819519, -1.100000023841858, 0.1839999556541443, 0.8239999413490295, -0.1640000343322754, -0.6399999856948853, -1.24399995803833, -0.9359999895095825 ], [ 0.0, -0.11599999666213989, 0.2720000147819519, 0.0, 0.18400001525878906, 0.8239999413490295, 0.0, -0.6399999856948853, -1.24399995803833 ], [ 0.0, 0.0, 0.0, 0.2720000147819519, -1.100000023841858, 0.0, 0.8239999413490295, -0.16400009393692017, 
```
...[skipping 6946 bytes](etc/1.txt)...
```
    78E-4, 2.384185791015625E-4, 0.0, 0.0, -2.384185791015625E-4 ], [ 1.423358917236328E-4, 1.6951560974121094E-4, -1.8477439880371094E-6, -2.8014183044433594E-5, 1.2981891632080078E-4, 1.1837482452392578E-4, 0.0, 0.0, -2.384185791015625E-4 ], [ 0.0, 1.423358917236328E-4, 2.2912025451660156E-4, 1.1920928955078125E-4, 9.119510650634766E-5, -1.0859966278076172E-4, -4.76837158203125E-4, -4.76837158203125E-4, -2.384185791015625E-4 ], [ 3.3158063888549805E-4, 3.838539123535156E-4, 1.1920928955078125E-4, 5.030632019042969E-5, -1.8477439880371094E-6, 2.384185791015625E-4, -3.470182418823242E-4, -1.2004375457763672E-4, 0.0 ], [ 3.0209869146347046E-4, 2.719759941101074E-4, 2.6226043701171875E-5, -9.608268737792969E-5, 1.6951560974121094E-4, -1.8477439880371094E-6, -3.8564205169677734E-4, -3.470182418823242E-4, -1.2004375457763672E-4 ], [ 2.384185791015625E-4, 3.0209869146347046E-4, 3.355741500854492E-5, -2.384185791015625E-4, 1.423358917236328E-4, 1.6951560974121094E-4, -2.384185791015625E-4, -1.4722347259521484E-4, -3.470182418823242E-4 ], [ 3.5762786865234375E-4, 1.1920928955078125E-4, 1.1920928955078125E-4, 3.355741500854492E-5, 1.4543533325195312E-4, 0.0, 1.6951560974121094E-4, 2.365708351135254E-4, -2.384185791015625E-4 ], [ 2.384185791015625E-4, 2.384185791015625E-4, 1.1920928955078125E-4, 6.368011236190796E-5, 1.5276670455932617E-4, -9.298324584960938E-5, 1.423358917236328E-4, 1.6951560974121094E-4, -2.402663230895996E-4 ], [ 2.384185791015625E-4, 2.384185791015625E-4, 1.1920928955078125E-4, -1.1920928955078125E-4, 1.828894019126892E-4, 3.9118528366088867E-4, -2.384185791015625E-4, -3.345012664794922E-4, -6.890296936035156E-5 ] ]
    Error Statistics: {meanExponent=-3.900262122498788, negative=28, min=-6.890296936035156E-5, max=-6.890296936035156E-5, mean=3.0377396830806026E-5, count=81.0, positive=45, stdDev=2.037233228692026E-4, zeros=8}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1440e-04 +- 1.1254e-04 [0.0000e+00 - 4.7684e-04] (162#)
    relativeTol: 1.9687e-01 +- 3.9745e-01 [1.2553e-06 - 1.0000e+00] (122#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1440e-04 +- 1.1254e-04 [0.0000e+00 - 4.7684e-04] (162#), relativeTol=1.9687e-01 +- 3.9745e-01 [1.2553e-06 - 1.0000e+00] (122#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 6.0930 +- 1.1901 [4.8845 - 10.7266]
    Learning performance: 4.0485 +- 0.6433 [3.5565 - 8.6605]
    
```

