AgriVision AI — Complete Implementation Roadmap
We will complete every requirement in STEPS by SIR.txt, including the optional features, while preserving the working classifier, FAISS index and Streamlit application. We will finish the required functionality first, then add optional features and final deployment.
Submission deadline: 30 October 2026.
Project completion plan

Phase 1 — Verify existing components
Mostly implemented
- Check dataset preparation and train/test split.
- Verify classifier artifacts and metrics.
- Test image prediction and disease-information mapping.
- Verify Streamlit history and clear-history behavior.
- Record the current working baseline before modifying code.




Phase 2 — Complete the RAG system
Integration needed
- Check disease knowledge documents for all supported classes.
- Integrate FAISS retrieval into the Streamlit app.
- Add a question box for disease-related questions.
- Display retrieved information and its source.
- Test symptom, treatment and prevention questions.




Phase 3 — Add multimodal GenAI
Pending
- Connect a multimodal model capable of processing images and text.
- Prefer Gemini API inference for the current 8 GB RAM laptop.
- Send the uploaded leaf image with a controlled prompt.
- Request descriptions only of symptoms visibly supported by the image.
- Handle API failures and unsupported image inputs.




Phase 4 — Integrate the complete advisory pipeline
Pending
Connect the components in this order:
Uploaded leaf image
MobileNetV3 classifier + visual description
FAISS retrieval + agricultural knowledge
Farmer-friendly advisory

The final answer must include crop, possible disease, visible symptoms, confidence, preventive actions and when to consult an agricultural expert.




Phase 5 — Add uncertainty and safety handling
Pending
- Show prediction confidence without treating it as a guaranteed probability of correctness.
- Detect low-confidence or ambiguous predictions using a validated threshold.
- Provide an inconclusive-result message when appropriate.
- Avoid asserting symptoms that the image does not show.
- Recommend expert inspection when evidence is insufficient.




Phase 6 — Implement optional local-language output
Optional in specification
- Add English, Kannada and Hindi selection.
- Translate the final advisory while preserving disease names and technical meaning.
- Test agricultural terms and treatment instructions for translation errors.
- Keep English as the default.




Phase 7 — Complete evaluation
Partially implemented
- Report accuracy, precision, recall and F1.
- Verify per-class performance and the confusion matrix.
- Test images from outside the training dataset.
- Evaluate visual-description accuracy and grounding.
- Evaluate whether advisory statements match retrieved knowledge.
- Document limitations, incorrect predictions and failure cases.




Phase 8 — Final UI, testing and deployment
Local app exists
- Improve layout, loading states and error messages.
- Show prediction history and allow it to be cleared.
- Add API-key configuration through environment variables or Streamlit secrets.
- Test the complete image-to-advisory workflow.
- Prepare setup instructions, requirements and a README.
- Deploy to a suitable hosting platform if feasible and verify the deployed application.



Timeline to submission
The dates below are our proposed schedule, not dates specified by your faculty.
Oct 13–15

Phases 1–2
Verify existing code and integrate the RAG question-answer feature.



Oct 16–19

Phases 3–4
Integrate the multimodal model and build the final advisory workflow.



Oct 20–22

Phases 5–6
Add uncertainty handling and English/Kannada/Hindi output.



Oct 23–26

Phase 7
Evaluate classification, external images, visual descriptions and advisory quality.



Oct 27–29

Phase 8
Complete integration tests, deployment if feasible, documentation and screenshots.



Oct 30

Submission
Verify the final files and prepare the project demonstration.



Completion criteria
We will consider the project ready only when these checks pass:
Project completion tracker
0/13 checked



Existing foundation
[ ] Classifier loads and predicts all supported classes
[ ] Metrics and evaluation artifacts are present
[ ] Streamlit image upload and prediction history work

RAG
[ ] FAISS retrieval is used by the application
[ ] Disease questions return relevant knowledge and sources

Multimodal GenAI
[ ] Image model describes visible symptoms
[ ] Final advisory combines image analysis and retrieved knowledge

Reliability
[ ] Uncertainty and inconclusive predictions are handled
[ ] Advice includes preventive actions and expert-referral guidance

Optional features
[ ] English, Kannada and Hindi outputs are tested

Submission
[ ] Evaluation report and failure cases are documented
[ ] README and setup instructions are complete
[ ] End-to-end app test and deployment status are recorded