
  /* ============================================================
    JAVASCRIPT
    WHY: Three jobs:
    1. setToggle()  — handles Yes/No toggle button state
    2. predict()    — collects all form values, sends to backend
    3. showResult() — renders the result panel
  ============================================================ */

    // ============================================================
    // 1. TOGGLE BUTTON LOGIC
    // WHY: Each Yes/No pair needs to know which is active.
    // We find the parent toggle-group and reset all siblings,
    // then mark the clicked one active.
    // ============================================================
    function setToggle(fieldId, value, clickedBtn) {
      // Store the value in the hidden input
      document.getElementById(fieldId).value = value;

      // Find all buttons in this toggle group
      const group = clickedBtn.parentElement;
      const buttons = group.querySelectorAll(".toggle-btn");

      // Reset all buttons in this group
      buttons.forEach(btn => {
        btn.classList.remove("active-yes", "active-no");
      });

      // Mark the clicked one
      if (value === 1) {
        clickedBtn.classList.add("active-yes");
      } else {
        clickedBtn.classList.add("active-no");
      }
    }

    // ============================================================
    // 2. PREDICT FUNCTION
    // WHY: Collects every field value, packages into a JSON object,
    // sends it to our FastAPI backend via fetch().
    // The field names MUST match exactly what the model was trained on.
    // ============================================================
    async function predict() {
      const btn     = document.getElementById("predictBtn");
      const btnText = document.getElementById("btnText");
      const spinner = document.getElementById("btnSpinner");

      // Show loading state
      btn.disabled   = true;
      btnText.style.display  = "none";
      spinner.style.display  = "inline";

      // Hide any previous result
      document.getElementById("resultPanel").classList.remove("show");

      // Collect all 32 feature values from the form
      // Each key matches exactly the column name the model was trained on
      const patientData = {
        Age:                       parseFloat(document.getElementById("Age").value),
        Gender:                    parseFloat(document.getElementById("Gender").value),
        Ethnicity:                 parseFloat(document.getElementById("Ethnicity").value),
        EducationLevel:            parseFloat(document.getElementById("EducationLevel").value),
        BMI:                       parseFloat(document.getElementById("BMI").value),
        Smoking:                   parseFloat(document.getElementById("Smoking").value),
        AlcoholConsumption:        parseFloat(document.getElementById("AlcoholConsumption").value),
        PhysicalActivity:          parseFloat(document.getElementById("PhysicalActivity").value),
        DietQuality:               parseFloat(document.getElementById("DietQuality").value),
        SleepQuality:              parseFloat(document.getElementById("SleepQuality").value),
        FamilyHistoryAlzheimers:   parseFloat(document.getElementById("FamilyHistoryAlzheimers").value),
        CardiovascularDisease:     parseFloat(document.getElementById("CardiovascularDisease").value),
        Diabetes:                  parseFloat(document.getElementById("Diabetes").value),
        Depression:                parseFloat(document.getElementById("Depression").value),
        HeadInjury:                parseFloat(document.getElementById("HeadInjury").value),
        Hypertension:              parseFloat(document.getElementById("Hypertension").value),
        SystolicBP:                parseFloat(document.getElementById("SystolicBP").value),
        DiastolicBP:               parseFloat(document.getElementById("DiastolicBP").value),
        CholesterolTotal:          parseFloat(document.getElementById("CholesterolTotal").value),
        CholesterolLDL:            parseFloat(document.getElementById("CholesterolLDL").value),
        CholesterolHDL:            parseFloat(document.getElementById("CholesterolHDL").value),
        CholesterolTriglycerides:  parseFloat(document.getElementById("CholesterolTriglycerides").value),
        MMSE:                      parseFloat(document.getElementById("MMSE").value),
        FunctionalAssessment:      parseFloat(document.getElementById("FunctionalAssessment").value),
        MemoryComplaints:          parseFloat(document.getElementById("MemoryComplaints").value),
        BehavioralProblems:        parseFloat(document.getElementById("BehavioralProblems").value),
        ADL:                       parseFloat(document.getElementById("ADL").value),
        Confusion:                 parseFloat(document.getElementById("Confusion").value),
        Disorientation:            parseFloat(document.getElementById("Disorientation").value),
        PersonalityChanges:        parseFloat(document.getElementById("PersonalityChanges").value),
        DifficultyCompletingTasks: parseFloat(document.getElementById("DifficultyCompletingTasks").value),
        Forgetfulness:             parseFloat(document.getElementById("Forgetfulness").value),
      };

      try {
        // WHY: fetch() sends an HTTP POST request to our FastAPI backend
        // running at localhost:8000. This is how frontend talks to backend.
        // We'll set up this URL when we build the backend.
        const response = await fetch("https://alzai.up.railway.app/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patientData)
        });

        if (!response.ok) throw new Error("Server error");

        const result = await response.json();
        showResult(result);

      } catch (error) {
        // WHY: Always handle errors gracefully — the backend might
        // not be running, or there could be a network issue
        alert("Could not connect to backend. Make sure app.py is running.\n\nError: " + error.message);
      }

      // Restore button
      btn.disabled          = false;
      btnText.style.display = "inline";
      spinner.style.display = "none";
    }

    // ============================================================
    // 3. SHOW RESULT
    // WHY: Takes the backend response and renders it visually.
    // Backend will return: { prediction: 0|1, prob_healthy: 0.97, prob_sick: 0.03 }
    // ============================================================
    function showResult(result) {
      const panel    = document.getElementById("resultPanel");
      const icon     = document.getElementById("resultIcon");
      const title    = document.getElementById("resultTitle");
      const subtitle = document.getElementById("resultSubtitle");
      const probH    = document.getElementById("probHealthy");
      const probS    = document.getElementById("probSick");
      const confText = document.getElementById("confidenceText");
      const confBar  = document.getElementById("confidenceBar");

      if (result.prediction === 1) {
        // POSITIVE — Alzheimer's detected
        panel.className = "result-panel show result-positive";
        icon.textContent    = "⚠️";
        title.textContent   = "Alzheimer's Risk Detected";
        title.className     = "font-serif text-3xl mb-2 text-red-400";
        subtitle.textContent = "The model indicates elevated risk. Please proceed with specialist referral.";
        confBar.style.background = "#ef4444";
        confBar.style.width      = (result.prob_sick * 100).toFixed(1) + "%";
        confText.textContent     = (result.prob_sick * 100).toFixed(1) + "%";
      } else {
        // NEGATIVE — No Alzheimer's
        panel.className = "result-panel show result-negative";
        icon.textContent    = "✅";
        title.textContent   = "No Alzheimer's Detected";
        title.className     = "font-serif text-3xl mb-2 text-teal-light";
        subtitle.textContent = "The model finds no significant indicators of Alzheimer's disease.";
        confBar.style.background = "#0d9488";
        confBar.style.width      = (result.prob_healthy * 100).toFixed(1) + "%";
        confText.textContent     = (result.prob_healthy * 100).toFixed(1) + "%";
      }

      probH.textContent = (result.prob_healthy * 100).toFixed(1) + "%";
      probS.textContent = (result.prob_sick    * 100).toFixed(1) + "%";

      // Smooth scroll to result
      panel.scrollIntoView({ behavior: "smooth", block: "center" });
    }
