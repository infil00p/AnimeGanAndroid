package org.infil00p.animegangallerydemo

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.widget.TextView
import org.infil00p.animegangallerydemo.databinding.ActivityMainBinding
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var mainBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        var external = this.getExternalFilesDir(null)

        var handler = AssetHandler(this)

        binding.getImage.setOnClickListener {
            try {
                val i = Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                )
                startActivityForResult(i, 0)
            } catch (e: Exception) {

            }
        }

        binding.doPredict.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : String
                if (external != null) {
                    predictVal = startPredict(byteBuffer, external.absolutePath,
                        mainBitmap!!.height, mainBitmap!!.width)
                }
            }
        }

        binding.doPredictWithGPU.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : String
                if(external != null) {
                    predictVal = startPredictWithGPU(
                        byteBuffer, external.absolutePath,
                        mainBitmap!!.height, mainBitmap!!.width)
                }
                
            }
        }

    }


    override fun onActivityResult(reqCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(reqCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            val imageUri = data!!.data
            val imageStream = contentResolver.openInputStream(imageUri!!)
            val exifStream = contentResolver.openInputStream(imageUri)
            val exif = ExifInterface(exifStream!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val rotMatrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotMatrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotMatrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotMatrix.postRotate(270f)
            }
            val selectedImage = BitmapFactory.decodeStream(imageStream)
            val rotatedBitmap = Bitmap.createBitmap(
                selectedImage, 0, 0,
                selectedImage.width, selectedImage.height,
                rotMatrix, true
            )

            // This is really important
            mainBitmap = rotatedBitmap

            runOnUiThread {
                binding.imageView.setImageBitmap(rotatedBitmap);
            }

        } else {
        }
    }

    /**
     * A native method that is implemented by the 'animegangallerydemo' native library,
     * which is packaged with this application.
     */

    external fun startPredict(buffer: ByteBuffer, externalFilePath: String, height: Int, width: Int) : String


    external fun startPredictWithGPU(buffer: ByteBuffer, externalFilePath: String, height: Int, width: Int) : String


    companion object {
        // Used to load the 'animegangallerydemo' library on application startup.
        init {
            System.loadLibrary("animegangallerydemo")
        }
    }
}