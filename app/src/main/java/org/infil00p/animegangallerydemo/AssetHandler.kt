package org.infil00p.animegangallerydemo

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class AssetHandler internal constructor(var mCtx: Context) {
    var LOGTAG = "AssetHandler"

    inner class ModelFileInit internal constructor(
        var mModelName: String,
        var mDataDir: File,
        var mAssetManager: AssetManager,
        var mModelFiles: Array<String>
    ) {
        var mTopLevelFolder: File? = null
        var mModelFolder: File? = null

        @Throws(IOException::class)
        private fun InitModelFiles() {
            copyModelFiles()
        }

        @Throws(IOException::class)
        private fun copyFileUtil(
            files: Array<String>
        ) {
            // For this example, we're using the internal storage
            for (file in files) {
                val inputFile = mAssetManager.open("$mModelName/$file")
                var outFile: File
                val dir = File(mDataDir.toString() + "/" +  mModelName)
                outFile = File(dir, file)
                val out: OutputStream = FileOutputStream(outFile)
                val buffer = ByteArray(1024)
                var length: Int
                while (inputFile.read(buffer).also { length = it } != -1) {
                    out.write(buffer, 0, length)
                }
                inputFile.close()
                out.flush()
                out.close()
            }
        }

        @Throws(IOException::class)
        private fun copyModelFiles() {
            copyFileUtil(mModelFiles)
        }


        private fun createTopLevelDir() {
            mTopLevelFolder = File(mDataDir.absolutePath, mModelName)
            mTopLevelFolder!!.mkdir()
        }

        init {
            createTopLevelDir()
            InitModelFiles()
        }
    }

    @Throws(IOException::class)
    private fun Init() {
        val dataDirectory = mCtx.filesDir
        val assetManager = mCtx.assets



        val pyTorch = ModelFileInit(
            "pytorch",
            dataDirectory,
            assetManager,
            arrayOf(
                "animegan2.pt",
                "animegan2_nhwc.pt",
                "animegan_vulkan_nchw.pt",
                "animegan_vulkan_nhwc.pt"
            )
        )


    }

    init {
        try {
            Init()
        } catch (e: IOException) {
            Log.d(LOGTAG, "Unable to get models from storage")
        }
    }
}