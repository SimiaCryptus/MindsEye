# SimpleConvolutionLayer
## Matrix
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "370a9587-74a1-4959-b406-fa4500000437",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500000437",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.876,
          -0.608,
          1.392,
          1.628,
          1.764,
          -0.016,
          1.96,
          -0.688,
          0.816
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ -0.708 ], [ -0.88 ], [ -0.82 ] ],
    	[ [ -1.396 ], [ -0.312 ], [ -1.272 ] ],
    	[ [ 1.836 ], [ 1.196 ], [ 1.656 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.401247501373291 ], [ -1.4244803190231323 ], [ -3.523375988006592 ] ],
    	[ [ -2.741168737411499 ], [ 1.9173604249954224 ], [ 2.3060169219970703 ] ],
    	[ [ 2.0995678901672363 ], [ -3.065040111541748 ], [ 1.8640961647033691 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000438",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000438",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.876,
          -0.608,
          1.392,
          1.628,
          1.764,
          -0.016,
          1.96,
          -0.688,
          0.816
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
    	[ [ -0.708 ], [ -0.88 ], [ -0.82 ] ],
    	[ [ -1.396 ], [ -0.312 ], [ -1.272 ] ],
    	[ [ 1.836 ], [ 1.196 ], [ 1.656 ] ]
    ]
    Error: [
    	[ [ 4.986267083673113E-7 ], [ -3.1902313235576685E-7 ], [ 1.1993408044475018E-8 ] ],
    	[ [ -7.374114994185277E-7 ], [ 4.2499542241003496E-7 ], [ 9.219970706908498E-7 ] ],
    	[ [ -1.0983276421683286E-7 ], [ -1.1154174828220675E-7 ], [ 1.6470336938745334E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 3.6668e-07 +- 2.9185e-07 [1.1993e-08 - 9.2200e-07] (9#)
    relativeTol: 8.3476e-08 +- 6.1386e-08 [1.7020e-09 - 1.9991e-07] (9#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.8560e-07 +- 3.2317e-07 [0.0000e+00 - 1.9073e-06] (180#), relativeTol=3.7578e-08 +- 6.9224e-08 [0.0000e+00 - 3.9863e-07] (180#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.708 ], [ -0.88 ], [ -0.82 ] ],
    	[ [ -1.396 ], [ -0.312 ], [ -1.272 ] ],
    	[ [ 1.836 ], [ 1.196 ], [ 1.656 ] ]
    ]
    Inputs Statistics: {meanExponent=0.0013912327564568185, negative=6, min=1.656, max=1.656, mean=-0.07777777777777778, count=9.0, positive=3, stdDev=1.206529069298261, zeros=0}
    Output: [
    	[ [ -2.401247501373291 ], [ -1.4244803190231323 ], [ -3.523375988006592 ] ],
    	[ [ -2.741168737411499 ], [ 1.9173604249954224 ], [ 2.3060169219970703 ] ],
    	[ [ 2.0995678901672363 ], [ -3.065040111541748 ], [ 1.8640961647033691 ] ]
    ]
    Outputs Statistics: {meanExponent=0.3603987540287474, negative=5, min=1.8640961647033691, max=1.8640961647033691, mean=-0.5520301394992404, count=9.0, positive=4, stdDev=2.386354226142133, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.708 ], [ -0.88 ], [ -0.82 ] ],
    	[ [ -1.396 ], [ -0.312 ], [ -1.272 ] ],
    	[ [ 1.836 ], [ 1.196 ], [ 1.656 ] ]
    ]
    Value Statistics: {meanExponent=0.0013912327564568185, negative=6, min=1.656, max=1.656, mean=-0.07777777777777778, count=9.0, positive=3, stdDev=1.206529069298261, zeros=0}
    Implemented Feedback: [ [ 1.7639999389648438, -0.016000032424926758, 0.0, -0.6880000829696655, 0.8159998655319214, 0.0, 0.0, 0.0, 0.0 ], [ 1.628000020980835, 1.7639999389648438, -0.016000032424926758, 1.9600000381469727, -0.6880000829696655, 0.8159999847412109, 0.0, 0.0, 0.0 ], [ 0.0, 1.628000020980835, 1.7639999389648438, 0.0, 1.9600000381469727, -0.6880000829696655, 0.0, 0.0, 0.0 ], [ -0.6080000400543213, 1.3919999599456787, 0.0, 1.7639999389648438, -0.016000032424926758, 0.0, -0.687999963760376, 0.8159999847412109, 0.0 ], [ -1.876000165939331, -0.6080000400543213, 1.3919999599456787, 1.628000020980835, 1.7639999389648438, -0.016000032424926758, 1.9600000381469727, -0.687999963760376, 0.8159999847412109 ], [ 0.0, -1.876000165939331, -0.6080000400543213, 0.0, 1.628000020980835, 1.7639999389648438, 0.0, 1.9600000381469727, -0.687999963760376 ], [ 0.0, 0.0, 0.0, -0.6079999208450317, 1.3919999599456787, 0.0, 1.7639999389648438, -0.016000032424926758, 0.
```
...[skipping 6896 bytes](etc/1.txt)...
```
    37361907958984E-5, -5.40614128112793E-4, 4.76837158203125E-4, -4.76837158203125E-4, -0.001430511474609375, 0.0 ], [ -2.4139881134033203E-4, 6.253421306610107E-4, -5.692243576049805E-4, 1.977086067199707E-4, -8.237361907958984E-5, 2.938508987426758E-4, -4.76837158203125E-4, -4.76837158203125E-4, 0.0 ], [ 0.0, 7.12275505065918E-4, -3.2833218574523926E-4, 3.5762786865234375E-4, -5.175471305847168E-4, 2.752542495727539E-4, -4.76837158203125E-4, -4.76837158203125E-4, -3.5762786865234375E-4 ], [ -1.7917156219482422E-4, 2.999305725097656E-4, -4.76837158203125E-4, 2.9295682907104492E-5, -4.500150680541992E-4, 0.0, -2.015829086303711E-4, -4.214048385620117E-4, -3.5762786865234375E-4 ], [ -3.4159421920776367E-4, 5.360841751098633E-4, -6.537437438964844E-4, 2.3543834686279297E-4, -3.2833218574523926E-4, 1.4603137969970703E-4, -1.5991926193237305E-4, -2.015829086303711E-4, -3.0219554901123047E-4 ], [ -4.76837158203125E-4, 3.7366151809692383E-4, -1.7917156219482422E-4, 2.384185791015625E-4, -2.4139881134033203E-4, 1.4850497245788574E-4, -4.76837158203125E-4, -1.5991926193237305E-4, -2.015829086303711E-4 ], [ 0.0, 7.152557373046875E-4, -2.384185791015625E-4, 1.7845630645751953E-4, -2.961158752441406E-4, 0.0, 1.4850497245788574E-4, -0.0010460615158081055, -1.1920928955078125E-4 ], [ -4.76837158203125E-4, 4.76837158203125E-4, 0.0, 1.6033649444580078E-5, -2.9838085174560547E-4, 5.383491516113281E-4, -4.7981739044189453E-4, -8.051693439483643E-4, -9.238719940185547E-5 ], [ -4.76837158203125E-4, 7.152557373046875E-4, 0.0, 3.5762786865234375E-4, -2.2238492965698242E-4, 2.976655960083008E-4, 0.0, -2.4139881134033203E-4, -2.09122896194458E-4 ] ]
    Error Statistics: {meanExponent=-3.519361841800626, negative=46, min=-2.09122896194458E-4, max=-2.09122896194458E-4, mean=-1.0987306818550016E-4, count=81.0, positive=25, stdDev=3.952834878233395E-4, zeros=10}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4274e-04 +- 2.5077e-04 [0.0000e+00 - 1.4305e-03] (162#)
    relativeTol: 1.8396e-01 +- 3.8666e-01 [9.9746e-06 - 1.0000e+00] (120#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.4274e-04 +- 2.5077e-04 [0.0000e+00 - 1.4305e-03] (162#), relativeTol=1.8396e-01 +- 3.8666e-01 [9.9746e-06 - 1.0000e+00] (120#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.6324 +- 0.9083 [4.9928 - 11.0885]
    Learning performance: 3.8368 +- 0.3575 [3.5423 - 5.5514]
    
```

