# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:76](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L76) executed in 0.10 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "606ba555-6e2c-49b8-b469-f6ba00000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/606ba555-6e2c-49b8-b469-f6ba00000002",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          -0.5384950684661942,
          0.16298019098851269,
          -0.6985456722742407,
          -0.5046770687347513,
          0.05899118444412643,
          -0.9053673259717632,
          -8.348037885739235E-4,
          0.43455099717134726,
          0.28243279653182785,
          0.6453526646763723,
          0.1492571211359619,
          0.1789613626336386,
          0.7266969758787198,
          -0.6443533982447911,
          -0.1442520015562243,
          0.6417846592056813,
          -0.5546150085087989,
          -0.059307650932010736,
          0.4881322475608878,
          -0.21590180945807091,
          0.039135507485025256,
          0.016176042383706513,
          0.6323810560762222,
          0.42393896329672676,
          -0.7883342778003009,
          0.5571991382084305,
          -0.1796773834168912,
          -0.39703631353431157,
          -0.9828068936587455,
          0.4556559505427098,
          -0.8895600818916587,
          0.9236799491998458,
          0.8603273047243949,
          -0.41255222231614597,
          -0.5781208456232161,
          0.4155095634651913,
          0.1593116736548552,
          0.8457962397271883,
          0.3839678218459939,
          0.36703535292695877,
          0.20123270535190407,
          0.6817490182994315,
          -0.9667115959531958,
          -0.6042616921104185,
          0.02031330151725408,
          -0.7642569533753971,
          0.957848406547466,
          -0.8267893214155424,
          -0.5827036454166816,
          0.5884071475062174,
          0.02979681238093712,
          -0.7023637152917963,
          -0.8638715910674721,
          -0.6686256531815984,
          -0.898572908566712,
          0.2210393987057686,
          0.3899676327952868,
          0.28829066147996274,
          0.3352825637184722,
          0.8300951533132384,
          -0.284121603299585,
          0.3506821937932456,
          0.3981012578308216,
          -0.9956666080761183,
          0.7118798455808746,
          0.5919734401
```
...[skipping 1282 bytes](etc/1.txt)...
```
    917107073567,
          0.9652508328659604,
          0.444297113729492,
          -0.17091986125362957,
          -0.05468748863417594,
          0.7106611967541039,
          0.6222625186580721,
          -0.20410291087435084,
          -0.3029285498193415,
          0.965444506928087,
          -0.11225609338154885,
          0.5862977314172733,
          0.9868581662049705,
          0.8590949364173903,
          -0.3075014318931466,
          -0.4464362710007106,
          -0.03639644446747292,
          0.22046345753515095,
          0.5422105521195095,
          0.8095588935818594,
          -0.29442092332160796,
          -0.010636980103514437,
          -0.987967113651216,
          0.2293771030975571,
          0.9259190120368599,
          0.45932023341560213,
          0.19777642141424145,
          0.30856551879336513,
          0.9091981827785907,
          0.06953227144686958,
          -0.7853875256329685,
          -0.06341301039300329,
          0.513332467540178,
          0.5231821616540866,
          0.5300647140239185,
          -0.2751311371761791,
          -0.3547461708118913,
          0.7503831135756425,
          0.32722379011208713,
          0.09241427622497023,
          0.5522609814948931,
          -0.6847107241171826,
          0.820833170869091,
          -0.22788591117802204,
          -0.622528345449771,
          0.19317975745199956,
          0.5071348490821235,
          -0.8897504308760007,
          0.6792187091032169,
          -0.7649727232646963,
          0.6256045104047812,
          0.0415128604204289,
          0.12687349407262083,
          0.731514388227744,
          0.16010644998680768,
          0.5220659011190938,
          -0.5474467337922078,
          0.36743610873884736,
          -0.5495408961744279,
          0.6896253036918667,
          0.5733587047159252,
          0.6968549186711059,
          0.12372197417773356,
          -0.3165704627096113,
          0.17277496911553114,
          -0.15799141693790442,
          0.3456545790384504,
          -0.5230617628867871,
          -0.5449980325191315,
          0.870186616818942,
          0.69879373055874,
          -0.44412455936715367,
          0.06709565358007796,
          0.8358181208471394,
          0.7866214602685757,
          0.639265932569334
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/606ba555-6e2c-49b8-b469-f6ba00000002
    Inputs: [[ [ [ 0.2586310236690261,-1.23636874308623,-1.312622146594748,1.9435170724426878,1.1677284069841836,0.940593653892758,-1.695905993936265 ],[ -1.0542892269105733,-0.010491856893660056,-1.754544825585803,-0.6523763633651836,-1.5176944230848997,0.9087339701542438,-1.2038077400228921 ],[ -1.6782754717875825,-1.729837099140818,-1.539139809753988,0.621123567800026,-1.136335418323282,1.1066111062019268,-0.21802944119836765 ] ],[ [ 1.1087065388028816,-0.7255540935767226,-1.6197379835247845,-1.1600932979350556,-1.3753125167555784,1.6092583571081462,1.1862809268796766 ],[ 1.4372144732361316,0.7511540326100219,0.8955629356451058,-1.9698828610675831,0.6302011901963511,0.9201067206415576,0.06633148550376822 ],[ 1.2666178705993945,0.19445391065060047,-0.5580786470557695,1.3823228420828033,1.0296535584996578,-0.2533382158174944,-1.9734045896926418 ] ],[ [ 1.144303081314423,-1.1291011827282311,-1.0468305974899947,1.578107815483305,0.8080490575392525,0.20404662763554393,-0.19748758306222358 ],[ 1.2255762295827757,-0.6539378389965176,1.8193561140426762,0.2867041241926529,-1.7256183135485648,-1.9585088390517518,-1.7976354700393706 ],[ -0.0794671272407994,0.5637176656681464,0.8586440514975906,1.4525086016688142,0.0735244193691762,-1.5627665496005965,0.17875989841696782 ] ] ]]
    Outputs: [ [ [ -5.138055199405869,7.868577439026687,-1.5068571900176244,-2.4671306959357446,-3.7609744575023627,-4.943569728618535,3.696873976521377 ],[ -6.086506716466648,-2.7976369714022513,-8.331320442601205,-6.388242529516216,2.129092816270371,0.058303986847135905,-0.32096774925027327 ],[ 1.5689946122926417,0.9203734715774939,-4.559600464268553,4.75818263940399,1.8495519341781375,-1.0679473775686026,-1.3509728707735005 ] ],[ [ 2.8767386459787376,-4.8854413211350805,3.6062725907536572,-2.114445604557639,-0.21866774707339864,-1.1781054868362482,0.8693840980127169 ],[ 6.866052188540131,-5.068079494592255,7.509083046550292,-1.4004609020479135,0.9621239850695344,1.7596423161196293,-2.184
```
...[skipping 1616 bytes](etc/2.txt)...
```
    827757,0.0,1.2666178705993945,-0.0794671272407994,0.0,0.0,0.0,... ],[ -1.0542892269105733,1.4372144732361316,1.2255762295827757,-1.6782754717875825,1.2666178705993945,-0.0794671272407994,0.0,0.0,... ],[ 0.0,-1.0542892269105733,1.4372144732361316,0.0,-1.6782754717875825,1.2666178705993945,0.0,0.0,... ],[ 1.1087065388028816,1.144303081314423,0.0,1.4372144732361316,1.2255762295827757,0.0,1.2666178705993945,-0.0794671272407994,... ],[ 0.2586310236690261,1.1087065388028816,1.144303081314423,-1.0542892269105733,1.4372144732361316,1.2255762295827757,-1.6782754717875825,1.2666178705993945,... ],[ 0.0,0.2586310236690261,1.1087065388028816,0.0,-1.0542892269105733,1.4372144732361316,0.0,-1.6782754717875825,... ],[ 0.0,0.0,0.0,1.1087065388028816,1.144303081314423,0.0,1.4372144732361316,1.2255762295827757,... ],[ 0.0,0.0,0.0,0.2586310236690261,1.1087065388028816,1.144303081314423,-1.0542892269105733,1.4372144732361316,... ],... ]
    [ [ 4.179487866906584E-10,-1.9165247167052257E-10,0.0,4.324887115103593E-10,1.3985879121491962E-10,0.0,0.0,0.0,... ],[ 3.231162004624366E-10,-2.6140423159404236E-11,-6.357416815205852E-10,3.036682016954728E-10,4.324887115103593E-10,1.3985879121491962E-10,0.0,0.0,... ],[ 0.0,-1.20973009387626E-10,-4.702296330094669E-10,0.0,3.036682016954728E-10,4.324887115103593E-10,0.0,0.0,... ],[ 2.1851631615277256E-10,-1.3468026693885804E-10,0.0,4.179487866906584E-10,2.5243673817954004E-10,0.0,-1.1600498339703336E-11,1.3985879121491962E-10,... ],[ 4.835518652157589E-10,-2.2557289369729006E-10,-5.787694767889207E-10,3.231162004624366E-10,4.179487866906584E-10,2.5243673817954004E-10,-1.404210081545898E-10,4.324887115103593E-10,... ],[ 0.0,3.9462655365696264E-11,-6.696621035473527E-10,0.0,3.231162004624366E-10,-2.6140423159404236E-11,0.0,3.036682016954728E-10,... ],[ 0.0,0.0,0.0,2.1851631615277256E-10,3.094089429112046E-10,0.0,-2.6140423159404236E-11,2.5243673817954004E-10,... ],[ 0.0,0.0,0.0,4.835518652157589E-10,2.1851631615277256E-10,-1.3468026693885804E-10,-1.20973009387626E-10,4.179487866906584E-10,... ],... ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=7.2615e-02 +- 3.0242e-01 [0.0000e+00 - 1.9734e+00] (11907#), relativeTol=4.4737e-01 +- 4.9722e-01 [1.0682e-12 - 1.0000e+00] (1862#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testLearning(DerivativeTester.java:250)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$2(DerivativeTester.java:75)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:76)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$7(LayerTestBase.java:101)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:100)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:66)
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
    	at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
    	at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:47)
    	at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
    	at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:325)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:78)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:57)
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



