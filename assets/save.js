
var form=document.getElementById("formId");
    	function savefile(){
            e.preventDefault();
        const question = document.getElementById('abc');
        var value = $("input[type=radio][name=flexRadioDefault]:checked").val();
        var s=""
        if (value=="Agree")
            s= "Yes,"+ question 
        elif(value=="Disagree")
            s="No,"+question
        elif(value=="Neutral")
            s=""
        
        let data = s + '|||';
        
        // Convert the text to BLOB.
        const textToBLOB = new Blob([data],  {type: 'text/plain'} );
        const sFileName = 'data.txt';	   // The file to save the data.

        let newLink = document.createElement("a");
        newLink.download = sFileName;

        if (window.webkitURL != null) 
            newLink.href = window.webkitURL.createObjectURL(textToBLOB);
        
        else 
            newLink.href = window.URL.createObjectURL(textToBLOB);
            newLink.style.display = "none";
            document.body.appendChild(newLink);
        
        newLink.click(); 
        }