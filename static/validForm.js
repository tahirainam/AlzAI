
(() => {
  'use strict';

  const form = document.querySelector("form");

  form.addEventListener("submit", (event) => {
    let isValid = true;

    // Clear old error styles
    form.querySelectorAll("input, select").forEach((field) => {
      field.classList.remove("border-red-500");
      const errorMsg = field.nextElementSibling;
      if (errorMsg && errorMsg.classList.contains("error-text")) {
        errorMsg.remove();
      }
    });

    // Validate required fields
    form.querySelectorAll("input[required], select[required]").forEach((field) => {
      if (!field.value || (field.type === "radio" && !form.querySelector(`input[name="${field.name}"]:checked`))) {
        isValid = false;
        field.classList.add("border-red-500");

        // Show error text (if not already present)
        if (!(field.nextElementSibling && field.nextElementSibling.classList.contains("error-text"))) {
          const error = document.createElement("p");
          error.className = "error-text text-red-500 text-sm mt-1";
          error.innerText = "This field is required.";
          field.insertAdjacentElement("afterend", error);
        }
      }
    });

    if (!isValid) {
      event.preventDefault();
      event.stopPropagation();
    }
  });
})();
