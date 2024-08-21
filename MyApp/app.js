// App.js
import React, { useState } from 'react';
import { View, Button, Image, ActivityIndicator, StyleSheet, Text } from 'react-native';
import axios from 'axios';
import { launchImageLibrary } from 'react-native-image-picker';

const App = () => {
  const [uploadUrl, setUploadUrl] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const getUploadUrl = async () => {
    try {
      const response = await axios.get('http://<YOUR_FLASK_SERVER>/get_upload_url');
      setUploadUrl(response.data.upload_url);
      setFileName(response.data.file_name);
    } catch (error) {
      console.error("Error getting upload URL: ", error);
    }
  };

  const uploadImage = async (imageUri) => {
    try {
      const response = await fetch(imageUri);
      const blob = await response.blob();
      
      await axios.put(uploadUrl, blob, {
        headers: {
          'Content-Type': 'image/jpeg',
        },
      });

      await checkProcessingStatus();
    } catch (error) {
      console.error("Error uploading image: ", error);
    }
  };

  const checkProcessingStatus = async () => {
    setIsLoading(true);
    const resultUrl = `http://<YOUR_FLASK_SERVER>/get_result/${fileName}`;
    try {
      while (true) {
        const response = await axios.get(resultUrl);
        if (response.data.processed_image_url) {
          setProcessedImageUrl(response.data.processed_image_url);
          setIsLoading(false);
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 5000)); // Poll every 5 seconds
      }
    } catch (error) {
      console.error("Error checking processing status: ", error);
      setIsLoading(false);
    }
  };

  const pickAndUploadImage = async () => {
    const result = await launchImageLibrary({ mediaType: 'photo' });

    if (result.didCancel) {
      console.log("User cancelled image picker");
    } else if (result.errorCode) {
      console.log("Image picker error: ", result.errorCode);
    } else {
      await getUploadUrl();
      await uploadImage(result.assets[0].uri);
    }
  };

  return (
    <View style={styles.container}>
      <Button title="Pick and Upload Image" onPress={pickAndUploadImage} />
      {isLoading && <ActivityIndicator size="large" color="#0000ff" />}
      {processedImageUrl && <Image source={{ uri: processedImageUrl }} style={styles.image} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  image: {
    width: 300,
    height: 300,
    marginTop: 20,
  },
});

export default App;
