# Orthanc DICOM Routing with AI Integration

A demo project showcasing DICOM study routing between Orthanc servers, with automatic forwarding to an AI inference server when studies stabilize.

## Terms
Orthanc Routing Server: The Orthanc server that routes studies to the AI server. Used to serve images to ODELIA viewer.
Orthanc AI Server: The Orthanc server that performs the AI inference. i.e. processes the images into format suitable for AI models and wraps AI results in DICOM SR/SC format. 
ODELIA Viewer: The viewer application that allows you to send studies to the AI server and view the results.
## Quick Test Guide
### 0. Start containers

This directory was historically a standalone repo (`orthanc-routing-example`); it is now part of the deployment. Build and run from the deployment root:

```bash
cd ..   # if you are inside orthanc/
docker compose up --build
```
### 1. Upload Studies to Routing Server

Use the Orthanc Explorer web interface at http://localhost:8000/app/explorer.html#upload to upload DICOM studies.

![Orthanc Explorer Upload](screenshots/orthanc-web-upload.png)  
*Web interface for uploading DICOM files*

---

### 2. Verify Study in Routing Server

1. Access Orthanc Explorer: http://localhost:8000/ui/app/index.html
![Routing Server Studies](screenshots/routing-server-studies.png)  
*Studies list in Orthanc Viewer*

![Original series](screenshots/original-series.png)  
*Original series in Orthanc Viewer*

---

### 3. Check AI Server Reception

1. Access AI Orthanc instance: http://localhost:8001/ui/app/index.html#/
![AI Server Studies](screenshots/ai-server-studies.png)  
*Received studies in AI Orthanc*

---

### Viewer Panel Integration

The ODELIA viewer includes a dedicated panel for AI integration that allows you to:
1. Send studies for AI processing
2. View AI results directly in the viewer interface

![Send to AI Panel](screenshots/send_to_ai_panel.jpg)
*AI processing panel in ODELIA Viewer*

![AI Result Viewer](screenshots/with_ai_result_viewer.jpg)
*Viewing AI results in ODELIA Viewer*

### 4. Check acquired AI results

1. Two new processed studies
![With AI results](screenshots/with_ai_results.jpg)
*Received mock AI results in ODELIA Viewer* 

2. Edited image sequence
![image sequence](screenshots/mock_ai_processed.jpg)
*Received mock visual AI result in Orthanc Viewer* 

3. Structure report generated
![SR sequence](screenshots/structured_report.jpg)
*Received mock SR AI result in Orthanc Viewer* 

4. View AI results in viewer
![AI Result Viewer](screenshots/with_ai_result_viewer.jpg)
*Viewing AI results in ODELIA Viewer*
