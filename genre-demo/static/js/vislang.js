
$(document).ready(function() { 
	var xhr = null;

	jQuery.fn.extend({
		showError: function(message){
			$(this).append('<p>' + message + '</p');
		},
		showInfo: function(message) {
			// $(this).append('<li class=\"alert alert-light text-smaller\">' + message + '</li>');
			$(this).append('<p>' + message + '</p>');
		}
	});

	/* Handle the submit behavior in the text2scene demo */
	$('#text2scene-form').on("submit", function(e){
		// Prevent multiple AJAX calls. 
		if(xhr && xhr.readyState != 4){ e.preventDefault(); return false;}

		if(this.checkValidity()){
			$('#loading-box').toggleClass("d-none");
			$('#output-image').toggleClass("d-none");
			$('#submit-button').toggleClass("disabled");

			// Submit the form using an asynchronous call.
	  		xhr = $.ajax({
	            url: 'text2scene-engine',
	            data: $(this).serializeArray(),
	            success: function(response){
	            	if(response != 0){
        				if(!("error" in response)) {
        					$('#loading-box').toggleClass("d-none");
        					$('#submit-button').toggleClass("disabled");
        					$('#output-image').attr("src", "data:image/png;base64," + response['output_image']);
        					$('#debug-str-box').html(response['debug_str'])
        					$('#output-image').toggleClass("d-none");

        				} else {
        					$('.flashes').showError(response['message']);
            				$('#loading-box').toggleClass("d-none");
            				$('#submit-button').toggleClass("disabled");
        				}
        			} else {
        				$('.flashes').showError("There was an error on the server side");
            			$('#loading-box').toggleClass("d-none");
            			$('#submit-button').toggleClass("disabled");
        			}
	            },
	            error: function(e) {
	           		$('.flashes').showError("There was an error on the server side");
            		$('#loading-box').toggleClass("d-none");
            		$('#submit-button').toggleClass("disabled");
	            }
	        });

		} else {
			$(this).addClass('was-validated');
		}
		e.preventDefault();
		return false;
	});

	/* End of the submit behavior in the text2scene demo */

	/* Handle the upload behavior in the gender demo */
	// Show the results on the screen.
	function showGenderDemoResults(response){
        if(response != 0){
        	if(!("error" in response)) {
        		// Empty error messages if any and hide the progress indicator.
            	$('#loading-box').toggleClass("d-none");
            	$('.flashes').empty()

        		// Place the input and output image content.
            	$("#g-input-image").attr("src", "data:image/jpeg;base64," + response['input_image']);
				// $("#g-output-image").attr("src", "data:image/jpeg;base64," + response['output_image']);
            	$('.flashes').showInfo(response['debug_str']);

            } else {
            	// Show an error and hide progress indicator.
            	$('.flashes').showError(response['message']);
            	$('#loading-box').toggleClass("d-none");
            }

        }else{
        	// Show an error and hide progress indicator.
            $('.flashes').showError('There was an error on the server side');
            $('#loading-box').toggleClass("d-none");
        }
    }

    // This only happens when the server throws a serious error e.g. not a 200 HTTP error code.
    function showGenderDemoError(response) {
		$('.flashes').showError('There was an error on the server side');
		$('#loading-box').toggleClass("d-none");
    }

    // Handle the paste url trigger.
	$("#form-file-paste").on("submit", function(e){
		// Prevent multiple AJAX calls. 
		if(xhr && xhr.readyState != 4){ xhr.abort(); e.preventDefault(); return false;}

		image_url = $("#image-input-url").val()

		// If the url is not a valid url then show error message.
		var regexp =  /^(?:(?:https?|ftp):\/\/)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:\/\S*)?$/;
		if(!regexp.test(image_url)) {
			// Avoid a browser POST request, empty messages and show error.
			e.preventDefault();
			$('.flashes').empty()
	  		$('.flashes').showError("Please enter a valid URL")
	  		return false;
	  	} else {
	  		// Clear any error messages and show loading bar.
	  		$('.flashes').empty()
	  		$('#loading-box').toggleClass("d-none");
	  		
	  		// Prepare data to be sent with form.
	  		var form_data = new FormData();
        	form_data.append('image_url', image_url);

        	// Submit the form using an asynchronous call.
	  		xhr = $.ajax({
	            url: 'simple-demo',
	            type: 'post',
	            data: form_data,
	            contentType: false,
	            processData: false,
	            success: showGenderDemoResults,
	            error: showGenderDemoError
	        });

	  		// Avoid a browser POST request.
	        e.preventDefault();
	        return false;
	  	}
	});

	// Handle the file upload trigger.
	$("#file-upload").on('change', function(e) {
		// Prevent multiple AJAX calls. 
		if(xhr && xhr.readyState != 4){ xhr.abort(); }

		// If the filename is too long then shorten.
		var fileName = $(this).val().split("\\").pop();
		if (fileName.length > 22) {
			fileName = fileName.substring(0, 10) + "..." + fileName.substring(fileName.length - 10, fileName.length);
		}

		// If the file size is bigger than 12MB then don't submit.
		var filesize = this.files[0].size / 1024 / 1024;
	  	if (filesize > 12) {
	  		var message =  "File size of " + filesize.toFixed(1) + "MB exceeds limit of 12MB";
	  		$('.flashes').empty();
	  		$('.flashes').showError(message);
	  	} else {
	  		// Show the filename in the input box.
			$(this).siblings(".custom-file-label-x").removeClass("form-control-placeholder").addClass("selected").html(fileName);

			// Clear any error messages and show loading bar.
			$('.flashes').empty()
			$('#loading-box').toggleClass("d-none");

			// Prepare data to be sent with form.
			var form_data = new FormData();
        	var files = this.files[0];
        	form_data.append('image', files);

			// Submit the form using an asynchronous call.
	  		xhr = $.ajax({
	            url: 'simple-demo',
	            type: 'post',
	            data: form_data,
	            contentType: false,
	            processData: false,
	            success: showGenderDemoResults,
	            error: showGenderDemoError
	        });
	  	}
	});

	/* Handle the click behavior in the text2scene demo */
	$("#submit-button").on("click", function(){
		$('#loading-box').removeClass("invisible").addClass("visible");
	});

	$(".coco-search-radios").on("click", function(){
		$("#coco-search-form").submit();
	});
})