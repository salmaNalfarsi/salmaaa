package com.example.objectdedection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PostProcessor
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.objectdedection.ml.AutoModel1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {



    lateinit var labels: List<String>
    val colors = listOf<Int>(
        Color.BLUE,Color.GREEN,Color.RED,Color.CYAN,Color.BLACK,
        Color.DKGRAY,Color.MAGENTA,Color.YELLOW,Color.RED)

    val paint = Paint()
    lateinit var imageProcessor:ImageProcessor
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model:AutoModel1


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)


        get_permission()
        labels = FileUtil.loadLabels(this,"mobilenet_objectdetection_labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300,300,ResizeOp.ResizeMethod.BILINEAR)).build()
        model = AutoModel1.newInstance(this)

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler= Handler(handlerThread.looper)

        imageView=findViewById(R.id.imageview)

        textureView = findViewById(R.id.textureview)
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {

            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                openCamera()

            }

            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {

            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false

            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap=textureView.bitmap!!


                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray!!
                val classes = outputs.classesAsTensorBuffer.floatArray!!
                val scores = outputs.scoresAsTensorBuffer.floatArray!!
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray!!

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width

                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f
                var x = 0

                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if (fl > 0.5) {
                        paint.color = colors[index]
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(
                            RectF(
                                locations[x + 1] * w,
                                locations[x] * h,
                                locations[x + 3] * w,
                                locations[x + 2] * h
                            ), paint
                        )
                        paint.style = Paint.Style.FILL
                        canvas.drawText(
                            labels[classes[index].toInt()] + fl.toString(),
                            locations[x + 1] * w,
                            locations[x] * h,
                            paint
                        )
                    }
                }
                imageView.setImageBitmap(mutable)
            }



        }
        cameraManager =getSystemService(Context.CAMERA_SERVICE)as CameraManager
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
    @SuppressLint("MissingPermission")
    fun openCamera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0],object:CameraDevice.StateCallback(){
            override fun onOpened(camera: CameraDevice) {
                cameraDevice= camera
                var surfaceTexture =textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var catureRequest= cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                catureRequest.addTarget(surface)
                cameraDevice.createCaptureSession(listOf(surface),object : CameraCaptureSession.StateCallback(){

                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(catureRequest.build(),null,null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                    }


                },handler)
            }

            override fun onDisconnected(camera: CameraDevice) {

            }

            override fun onError(camera: CameraDevice, error: Int) {

            }


        },handler)
    }
    fun get_permission() {
        if(ContextCompat.checkSelfPermission(this,android.Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA),101)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0]!=PackageManager.PERMISSION_GRANTED)
            get_permission()
    }
}




